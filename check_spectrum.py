import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import argparse

from dataset.kitti.parser import Parser
from modules.aug_model import AugModel

@torch.no_grad()
def spectrum_report(model, loader, device, max_per_class=5000, q=512):
    """Report the eigen-spectrum of each class's HDC covariance, and how many
    directions clear the paper's threshold at several choices of tau."""
    d = model.hd_dim
    model.eval()

    # ---- collect per-class hypervectors -------------------------------------
    buckets = {c: [] for c in range(model.num_classes)}
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device).view(-1)
        
        if x.shape[1] == 0:
            continue
            
        enc, indices, _ = model.encode(x)
        h = F.normalize(enc)
        labels = y[indices] if indices is not None else y
        valid = (labels >= 0) & (labels < model.num_classes)
        h, labels = h[valid], labels[valid]
        for c in labels.unique().tolist():
            if counts[c] >= max_per_class:
                continue
            hc = h[labels == c]
            take = min(hc.shape[0], max_per_class - counts[c])
            buckets[c].append(hc[:take].cpu())
            counts[c] += take
        if all(v >= max_per_class for v in counts.values()):
            break

    print(f"\nd = {d}   (mean eigenvalue if perfectly isotropic = R^2/d)")
    print(f"{'cls':>4} {'n':>6} {'rank':>5} {'eff_rank':>9} "
          f"{'lam1/mean':>10} {'top10%var':>10} "
          f"{'r@tau=d^.25':>12} {'r@tau=d^.125':>13} {'r@90%var':>9}")
    print("-" * 95)

    summary = {}
    for c in range(model.num_classes):
        if not buckets[c]:
            continue
        Y = torch.cat(buckets[c]).to(device).float()
        n = Y.shape[0]
        delta = Y - Y.mean(0)

        R = torch.quantile(delta.norm(dim=1), 0.99).item()
        qq = int(min(q, n - 1, d))
        _, S, _ = torch.pca_lowrank(delta, q=qq, center=False, niter=4)
        lam = ((S ** 2) / max(n - 1, 1)).cpu().numpy()

        trace = float(lam.sum())          # NB: truncated -- lower bound on true trace
        mean_eig_iso = (R ** 2) / d       # what the mean would be if isotropic

        # how many clear the threshold at various tau?
        def r_at(tau_exp):
            tau_sq = d ** (2 * tau_exp)
            thresh = tau_sq * (R ** 2) / d
            return int((lam >= thresh).sum())

        r_paper = r_at(0.25)              # tau = d^{1/4}  -> threshold R^2/sqrt(d)
        r_loose = r_at(0.125)             # tau = d^{1/8}  -> threshold R^2/d^{3/4}

        # variance-based rank (how many dirs to explain 90% of captured variance)
        cum = np.cumsum(lam) / max(trace, 1e-12)
        r_90 = int(np.searchsorted(cum, 0.90) + 1)

        # effective rank = exp(entropy of the normalized spectrum). A flat (isotropic)
        # spectrum has eff_rank ~ number of dirs; a peaked one has eff_rank ~ few.
        p = lam / max(trace, 1e-12)
        p = p[p > 0]
        eff_rank = float(np.exp(-(p * np.log(p)).sum()))

        lam1_ratio = float(lam[0] / max(mean_eig_iso, 1e-12))
        top10 = float(lam[:max(1, len(lam) // 10)].sum() / max(trace, 1e-12))

        print(f"{c:>4} {n:>6} {qq:>5} {eff_rank:>9.1f} "
              f"{lam1_ratio:>10.1f} {top10:>10.2%} "
              f"{r_paper:>12} {r_loose:>13} {r_90:>9}")

        summary[c] = dict(n=n, R=R, eff_rank=eff_rank, lam1_over_iso_mean=lam1_ratio,
                          r_paper=r_paper, r_loose=r_loose, r_90=r_90)

    # ---- verdict -------------------------------------------------------------
    r_papers = [v["r_paper"] for v in summary.values()]
    print("\n" + "=" * 95)
    if not r_papers:
        print("No classes collected.")
        return summary
    if max(r_papers) == 0:
        print("VERDICT: r = 0 for EVERY class at the paper's tau = d^{1/4}.")
        print("  The shape matrix is the identity. The 'ellipsoid' IS a ball, and the")
        print("  method will score IDENTICALLY to the prototype baseline.")
        print("  -> DO NOT run the gate-quality experiment yet. Lower tau first (see below).")
    elif min(r_papers) == 0:
        print(f"VERDICT: r = 0 for SOME classes (r per class: {r_papers}).")
        print("  Those classes degenerate to balls while others do not -- an inconsistent")
        print("  gate across classes. Lower tau so every class gets a non-trivial shape.")
    else:
        print(f"VERDICT: r > 0 for all classes (r per class: {r_papers}). ")
        print("  The covariance is anisotropic enough for the construction to bite.")
        print("  Proceed to the gate-quality experiment.")

    print(\"\"\"
IF YOU NEED TO LOWER tau:
  tau is a FREE PARAMETER. The paper sets tau = d^{1/4} to optimize a WORST-CASE
  volume bound (their Sec 2.2, balancing losses (ii) and (iii)) -- not to be
  well-calibrated on any particular real distribution. Lowering it admits more
  directions into the high-variance set:

      tau = d^{1/4}  -> threshold R^2 / d^{1/2},  r <= d^{1/2} = 100
      tau = d^{1/8}  -> threshold R^2 / d^{3/4},  r <= d^{3/4} = 1000

  Everything in the implementation is parameterized by `tau_exponent`, so this is a
  one-line change. The cost: the volume-competitiveness constant degrades (property
  (iii) bounds distortion by d^{d/tau^2}, which grows as tau shrinks). BE EXPLICIT
  ABOUT THIS IN THE PAPER -- you are trading a worst-case guarantee for practical
  anisotropy on real data, and a reviewer will ask.

  Alternative: keep the BINNING STRUCTURE (eigenvalues in {1, d}, which is what buys
  non-expansion and bounded distortion) but select r by a spectral criterion (e.g.
  the r_90 column above) instead of an absolute threshold. Same geometry, data-driven
  rank. This is a defensible modification and arguably the honest one.
\"\"\")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='/home/james/Research/SEE/DensityUnsupHyperLidar/logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model weights')
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to KITTI data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and architecture
    CONFIG_ARCH = "config/arch/senet-2048p.yml"
    CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
    
    with open(CONFIG_ARCH, 'r') as f:
        ARCH = yaml.safe_load(f)
    with open(CONFIG_LABELS, 'r') as f:
        DATA = yaml.safe_load(f)

    # Initialize Parser on valid sequence (08)
    dataset_parser = Parser(
        root=args.kitti_dir,
        train_sequences=DATA["split"]["valid"],
        valid_sequences=DATA["split"]["valid"],
        test_sequences=None,
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=8,
        workers=ARCH["train"]["workers"],
        gt=True,
        shuffle_train=False
    )
    dataloader = dataset_parser.get_train_set()

    # Load Model (using AugModel for compatibility with the old repo)
    num_classes = 17
    modeldir = os.path.dirname(args.pretrained_path)
    model = AugModel(ARCH, modeldir, 'rp', 0, 0, num_classes, device, subcluster_type='continuous')
    
    print(f"Loading pretrained model from {args.pretrained_path}...")
    try:
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.to(device)

    # Run Spectrum Diagnostic
    print("\nRunning spectrum report diagnostic...")
    spectrum_report(model, dataloader, device)

if __name__ == "__main__":
    main()

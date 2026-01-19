import torch
import yaml

from dataset.kitti.parser import Parser
from faster_mean_shift.mean_shift_cosine_gpu import estimate_bandwidth_binary, mean_shift_binary
from modules.HDC_utils import DensityModel
from modules.trainer import Trainer
from modules.ioueval import iouEval

import numpy as np
import torch.nn.functional as F

from unsup_main import train_extractor, train_hdc, test_hdc_model, test_hdc_model_debug

MODEL_DIR = "logs"
NU_DATA_DIR = "v1.0-mini"
DATA_DIR = "nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_EPOCHS = 10
MAX_HDC_EPOCHS = 10

HD_DIM = 5000

HDC_SUB_PATH = "logs/hdc_sub.pth"

def diag_similarity_dynamic_range(model, dataloader):
    device = model.device
    ranges = {c: [] for c in range(model.num_classes)}

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            labels = labels.to(device)

            enc, _, _ = model.encode(proj_in)
            enc = F.normalize(enc.to(torch.float))
            
            # flatten batch
            labels_flat = labels.view(-1)
            enc_flat = enc[:labels_flat.shape[0]]

            for c in range(model.num_classes):
                mask = labels_flat == c
                if mask.sum() == 0:
                    continue
                enc_c = enc_flat[mask]  # [N_c, D]
                sub_c = model.subclusters[model.subcluster_to_class == c]  # [S, D]

                # batched similarity
                sims = enc_c @ sub_c.T  # [N_c, S]
                diffs = sims.max(dim=1).values - sims.mean(dim=1)
                ranges[c].extend(diffs.cpu().tolist())

    print("\n[Similarity Dynamic Range]")
    for c, vals in ranges.items():
        if vals:
            vals = torch.tensor(vals)
            print(f"Class {c:2d} | mean range={vals.mean():.3f}")


def diag_subcluster_usage(model, dataloader):
    device = model.device
    usage = {c: torch.zeros(model.num_subclusters, device=device) for c in range(model.num_classes)}

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            labels = labels.to(device)

            enc, _, _ = model.encode(proj_in)
            enc = F.normalize(enc.to(torch.float))

            labels_flat = labels.view(-1)
            enc_flat = enc[:labels_flat.shape[0]]

            for c in range(model.num_classes):
                mask = labels_flat == c
                if mask.sum() == 0:
                    continue
                enc_c = enc_flat[mask]  # [N_c, D]
                sub_c = model.subclusters[model.subcluster_to_class == c]  # [S, D]

                sims = enc_c @ sub_c.T  # [N_c, S]
                max_indices = sims.argmax(dim=1)
                counts = torch.bincount(max_indices, minlength=model.num_subclusters)
                usage[c] += counts

    print("\n[Subcluster Utilization]")
    for c, counts in usage.items():
        if counts.sum() == 0:
            continue
        frac = counts / counts.sum()
        entropy = -(frac * torch.log(frac + 1e-9)).sum()
        print(f"Class {c:2d} | entropy={entropy:.2f} min={frac.min():.2%} max={frac.max():.2%}")


def diag_subcluster_separation(model, dataloader):
    device = model.device
    gaps = {c: [] for c in range(model.num_classes)}

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            labels = labels.to(device)

            enc, _, _ = model.encode(proj_in)
            enc = F.normalize(enc.to(torch.float))

            labels_flat = labels.view(-1)
            enc_flat = enc[:labels_flat.shape[0]]

            # compute all similarities at once
            all_subclusters = model.subclusters  # [C*S, D]
            all_sims = enc_flat @ all_subclusters.T  # [N, C*S]

            for c in range(model.num_classes):
                mask = labels_flat == c
                if mask.sum() == 0:
                    continue

                enc_c_sims = all_sims[mask]  # [N_c, C*S]

                # split into per-class subclusters
                per_class_sims = []
                for j in range(model.num_classes):
                    sc_mask = model.subcluster_to_class == j
                    per_class_sims.append(enc_c_sims[:, sc_mask].max(dim=1).values)
                per_class_sims = torch.stack(per_class_sims, dim=1)  # [N_c, C]

                # gap = correct class max - next best
                top2 = torch.topk(per_class_sims, 2, dim=1).values
                class_gap = top2[:, 0] - top2[:, 1]
                gaps[c].extend(class_gap.cpu().tolist())

    print("\n[Subcluster Separation Diagnostics]")
    for c, vals in gaps.items():
        if vals:
            vals = torch.tensor(vals)
            print(f"Class {c:2d} | mean gap={vals.mean():.3f} p05={vals.quantile(0.05):.3f}")


def diag_subcluster_coverage(model, dataloader, beta=0.6):
    device = model.device
    stats = {c: [] for c in range(model.num_classes)}

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            labels = labels.to(device)

            enc, _, _ = model.encode(proj_in)
            enc = F.normalize(enc.to(torch.float))

            labels_flat = labels.view(-1)
            enc_flat = enc[:labels_flat.shape[0]]

            all_subclusters = model.subclusters  # [C*S, D]
            all_sims = enc_flat @ all_subclusters.T  # [N, C*S]

            for c in range(model.num_classes):
                mask = labels_flat == c
                if mask.sum() == 0:
                    continue
                sc_mask = model.subcluster_to_class == c
                sims = all_sims[mask][:, sc_mask]
                stats[c].extend(sims.max(dim=1).values.cpu().tolist())

    print("\n[Subcluster Coverage Diagnostics]")
    for c, vals in stats.items():
        if vals:
            vals = torch.tensor(vals)
            print(
                f"Class {c:2d} | mean={vals.mean():.3f} "
                f"p05={vals.quantile(0.05):.3f} "
                f"p50={vals.median():.3f} "
                f"p95={vals.quantile(0.95):.3f} "
                f"<β={(vals < beta).float().mean():.2%}"
            )

def diag_init_subclusters(model, dataloader, bandwidth=None, quantile=0.2, max_samples_per_class=5000, max_subclusters=None, verbose=True,):
    """
    Diagnostic version of init_subclusters.
    Does NOT modify model.subclusters.
    """
    device = model.device
    if max_subclusters is None:
        max_subclusters = model.num_subclusters

    results = {}

    model.eval()
    with torch.no_grad():
        for class_id in range(model.num_classes):
            if verbose:
                print(f"\n[Class {class_id}] collecting embeddings")

            class_embeddings = []
            total = 0

            for batch_idx, (proj_in, _, labels, *_) in enumerate(dataloader):
                proj_in = proj_in.to(device)
                labels = labels.to(device).view(-1)

                enc, _, _ = model.encode(proj_in)
                enc = torch.sign(enc).cpu()

                mask = labels == class_id
                if torch.any(mask):
                    class_enc = enc[mask.cpu()]
                    class_embeddings.append(class_enc)
                    total += class_enc.shape[0]

                if total >= max_samples_per_class:
                    break

            if not class_embeddings:
                results[class_id] = {
                    "status": "NO_SAMPLES",
                    "num_samples": 0,
                }
                if verbose:
                    print("  ❌ no samples found")
                continue

            X = torch.cat(class_embeddings, dim=0)
            if X.shape[0] > max_samples_per_class:
                idx = torch.randperm(X.shape[0])[:max_samples_per_class]
                X = X[idx]

            X_np = X.numpy()

            if bandwidth is None:
                est_bw = estimate_bandwidth_binary(
                    X_np,
                    quantile=quantile,
                    n_samples=min(500, len(X_np))
                )
            else:
                est_bw = bandwidth

            if verbose:
                print(f"  samples={len(X_np)}  bandwidth={est_bw:.4f}")

            try:
                centers = mean_shift_binary(X_np, bandwidth=est_bw)
            except Exception as e:
                results[class_id] = {
                    "status": "MEAN_SHIFT_FAILED",
                    "error": str(e),
                    "num_samples": len(X_np),
                    "bandwidth": est_bw,
                }
                if verbose:
                    print(f"  ❌ mean shift failed: {e}")
                continue

            if len(centers) == 0:
                results[class_id] = {
                    "status": "NO_CLUSTERS",
                    "num_samples": len(X_np),
                    "bandwidth": est_bw,
                }
                if verbose:
                    print("  ❌ mean shift returned 0 clusters")
                continue

            centers = np.sign(centers)

            centers_t = torch.tensor(centers)
            unique_centers = torch.unique(centers_t, dim=0).shape[0]

            sims = torch.matmul(
                centers_t.to(torch.float),
                X.to(torch.float).T
            )
            sims = sims / X.shape[1]

            results[class_id] = {
                "status": "OK",
                "num_samples": len(X_np),
                "bandwidth": est_bw,
                "num_clusters_found": len(centers),
                "num_unique_clusters": unique_centers,
                "max_allowed_clusters": max_subclusters,
                "mean_similarity": sims.mean().item(),
                "p05_similarity": torch.quantile(sims.flatten(), 0.05).item(),
                "p95_similarity": torch.quantile(sims.flatten(), 0.95).item(),
                "degenerate": unique_centers == 1,
            }

            if verbose:
                print(
                    f"  clusters_found={len(centers)} "
                    f"unique={unique_centers} "
                    f"degenerate={unique_centers == 1}"
                )
                print(
                    f"  sim mean={results[class_id]['mean_similarity']:.4f} "
                    f"p05={results[class_id]['p05_similarity']:.4f}"
                )

    return results

def test_init(model, trainloader):
    diag = diag_init_subclusters(
        model,
        trainloader,
        quantile=0.2,
        max_samples_per_class=3000,
        verbose=True,
    )

    for c, r in diag.items():
        if r["status"] != "OK":
            print(f"Class {c}: {r}")

def subcluster_test_suite(model, dataloader):
    diag_subcluster_coverage(model, dataloader)
    diag_similarity_dynamic_range(model, dataloader)
    diag_subcluster_separation(model, dataloader)
    diag_subcluster_usage(model, dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        ARCH = yaml.safe_load(open("config/arch/senet-512.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_mini.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    DATA['split']['train'] = [61, 103, 553, 655, 757, 796, 916, 1077, 1094, 1100]
    ARCH["train"]["batch_size"] = 2

    parser = Parser(root=DATA_DIR,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    
    trainloader = parser.get_train_set()

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    # subcluster_test_suite(model, trainloader)
    test_init(model, trainloader)

if __name__=="__main__":
    main()
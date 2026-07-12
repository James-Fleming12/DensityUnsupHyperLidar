import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from modules.aug_model import AugModel
from dataset.kitti.parser import Parser

BINS = 20000
BIN_EDGES = np.linspace(-1.0, 1.0, BINS + 1)

def update_hists(is_correct, sims, hist_correct, hist_incorrect):
    is_correct = np.array(is_correct, dtype=bool)
    sims = np.array(sims, dtype=np.float32)
    
    c_sims = sims[is_correct]
    i_sims = sims[~is_correct]
    
    hc, _ = np.histogram(c_sims, bins=BIN_EDGES)
    hi, _ = np.histogram(i_sims, bins=BIN_EDGES)
    
    hist_correct += hc
    hist_incorrect += hi

def compute_metrics_from_hist(hist_correct, hist_incorrect):
    cum_correct = np.cumsum(hist_correct[::-1])[::-1]
    cum_incorrect = np.cumsum(hist_incorrect[::-1])[::-1]
    cum_total = cum_correct + cum_incorrect
    
    total_correct = cum_correct[0]
    total_incorrect = cum_incorrect[0]
    
    tpr = cum_correct / max(total_correct, 1)
    fpr = cum_incorrect / max(total_incorrect, 1)
    
    # AUC uses trapezoidal rule. FPR and TPR are decreasing as we sweep threshold from +1 to -1.
    # We must sort them ascending for np.trapz, or just take negative.
    # fpr[::-1] is ascending
    auroc = np.trapz(tpr[::-1], fpr[::-1])
    
    coverage = cum_total / max(cum_total[0], 1)
    precision = np.zeros_like(coverage)
    valid = cum_total > 0
    precision[valid] = cum_correct[valid] / cum_total[valid]
    
    # Filter out empty bins for clean plotting
    valid_plot = np.where(cum_total > 0)[0]
    if len(valid_plot) > 0:
        first_valid = valid_plot[0]
        coverage = coverage[first_valid:]
        precision = precision[first_valid:]
        
    return auroc, coverage, precision

def test_auroc_on_chunk(corruption_root, pretrained_path, yaml_labels, yaml_arch, device, corruption_name, output_dir):
    DATA = yaml.safe_load(open(yaml_labels, 'r'))
    ARCH = yaml.safe_load(open(yaml_arch, 'r'))
    
    try:
        parser_obj = Parser(root=corruption_root,
                            train_sequences=DATA["split"]["valid"],
                            valid_sequences=DATA["split"]["valid"],
                            test_sequences=None,
                            labels=DATA["labels"],
                            color_map=DATA.get("color_map", {}),
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=1,
                            workers=8,
                            gt=True,
                            shuffle_train=False)
    except Exception as e:
        print(f"Failed to load dataset for {corruption_name}: {e}")
        return
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False, num_workers=8)
    
    # Load model
    modeldir = os.path.dirname(pretrained_path)
    model = AugModel(ARCH, modeldir, 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
    model = model.to(device)
    model.eval()
    
    # Create K=1 Subclusters (class centroids from subclusters)
    hd_dim = model.classify.weight.shape[1]
    k1_subclusters = torch.zeros((17, hd_dim), device=device, dtype=model.classify.weight.dtype)
    for c_id in range(17):
        mask = model.subcluster_to_class == c_id
        if mask.sum() > 0:
            c_subs = model.subclusters[mask].float()
            k1_subclusters[c_id] = c_subs.mean(dim=0).to(k1_subclusters.dtype)
    
    h_corr_proto = np.zeros(BINS, dtype=np.int64)
    h_incorr_proto = np.zeros(BINS, dtype=np.int64)
    
    h_corr_k1 = np.zeros(BINS, dtype=np.int64)
    h_incorr_k1 = np.zeros(BINS, dtype=np.int64)
    
    h_corr_k64 = np.zeros(BINS, dtype=np.int64)
    h_incorr_k64 = np.zeros(BINS, dtype=np.int64)
    
    total_points = 0
    total_correct_points = 0
    
    print(f"\n--- Running AUROC Gate-Quality Test on {corruption_name} ---")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            proj_in = batch_data[0].to(device)
            oracle_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] == 0:
                continue
                
            enc_base, _, _ = model.encode(proj_in)
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            if not torch.any(valid_enc_mask):
                continue
                
            raw_base = F.normalize(enc_base[valid_enc_mask])
            active_oracle = oracle_labels.reshape(-1)[valid_enc_mask]
            
            # Predict
            prototypes = F.normalize(model.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)
            
            S_base = raw_base @ prototypes.T
            preds = S_base.argmax(dim=1)
            
            valid_labels_mask = (active_oracle > 0) & (active_oracle < 17) & (preds > 0)
            
            if not torch.any(valid_labels_mask):
                continue
                
            filtered_preds = preds[valid_labels_mask]
            filtered_oracle = active_oracle[valid_labels_mask]
            filtered_raw_base = raw_base[valid_labels_mask]
            
            is_correct = (filtered_preds == filtered_oracle).cpu().numpy()
            
            total_points += len(is_correct)
            total_correct_points += np.sum(is_correct)
            
            # 1. Prototype
            selected_proto = prototypes[filtered_preds]
            proto_sims = torch.sum(filtered_raw_base * selected_proto, dim=1).cpu().numpy()
            update_hists(is_correct, proto_sims, h_corr_proto, h_incorr_proto)
            
            # 2. Subcluster Similarity (K=1)
            k1_sims = torch.zeros(filtered_raw_base.shape[0], device=device)
            for c_id in torch.unique(filtered_preds):
                c_id_item = c_id.item()
                c_mask = (filtered_preds == c_id)
                c_encs = filtered_raw_base[c_mask]
                
                c_k1_proto = F.normalize(k1_subclusters[c_id_item].unsqueeze(0), dim=1)
                k1_sims[c_mask] = torch.sum(c_encs * c_k1_proto, dim=1)
            update_hists(is_correct, k1_sims.cpu().numpy(), h_corr_k1, h_incorr_k1)
            
            # 3. Subcluster Similarity (K=64)
            sub_sims = torch.zeros(filtered_raw_base.shape[0], device=device)
            for c_id in torch.unique(filtered_preds):
                c_id_item = c_id.item()
                c_mask = (filtered_preds == c_id)
                c_encs = filtered_raw_base[c_mask]
                c_sub_sims, _ = model.get_max_subcluster_similarity(c_encs, c_id_item, distance_sensitivity=1.0)
                sub_sims[c_mask] = c_sub_sims
            update_hists(is_correct, sub_sims.cpu().numpy(), h_corr_k64, h_incorr_k64)
            
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"Processed {batch_idx} frames...")
                
    if total_points == 0:
        print("No valid points found to evaluate.")
        return
        
    try:
        proto_auroc, cov_proto, prec_proto = compute_metrics_from_hist(h_corr_proto, h_incorr_proto)
        sub_k1_auroc, cov_k1, prec_k1 = compute_metrics_from_hist(h_corr_k1, h_incorr_k1)
        sub_k64_auroc, cov_k64, prec_k64 = compute_metrics_from_hist(h_corr_k64, h_incorr_k64)
        
        print("\n=== AUROC RESULTS ===")
        print(f"Total Points Evaluated: {total_points:,}")
        print(f"Base Accuracy of Pseudo-Labels: {total_correct_points / total_points:.2%}")
        print(f"AUROC (Prototype):          {proto_auroc:.4f}")
        print(f"AUROC (Subcluster K=1):     {sub_k1_auroc:.4f}")
        print(f"AUROC (Subcluster K=64):    {sub_k64_auroc:.4f}")
        print("=====================\n")
        
        plt.figure(figsize=(10, 6))
        plt.plot(cov_proto, prec_proto, label=f'Prototype (AUROC={proto_auroc:.3f})', linewidth=2)
        plt.plot(cov_k1, prec_k1, label=f'Subcluster K=1 (AUROC={sub_k1_auroc:.3f})', linewidth=2)
        plt.plot(cov_k64, prec_k64, label=f'Subcluster K=64 (AUROC={sub_k64_auroc:.3f})', linewidth=2)
        
        plt.title(f'Gate Quality: Precision vs Coverage ({corruption_name})', fontsize=14)
        plt.xlabel('Coverage (Fraction of points admitted)', fontsize=12)
        plt.ylabel('Precision (Accuracy of admitted points)', fontsize=12)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'precision_coverage_{corruption_name}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Precision-Coverage plot to {out_path}")
        
    except Exception as e:
        print(f"Could not calculate AUROC or generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AUROC Gate Quality Test")
    parser.add_argument('--corruptions', type=str, default="wet_ground,snow,beam_missing,motion_blur,incomplete_echo", help="Comma separated list of corruptions")
    parser.add_argument('--severity', type=int, default=3, help="Severity level")
    parser.add_argument('--kittic_dir', type=str, default='/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth')
    parser.add_argument('--output_dir', type=str, default='logs/auroc_tests')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_labels = 'config/labels/semantic-kitti-all.yaml'
    yaml_arch = 'config/arch/senet-2048p.yml'
    
    SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy', 4: 'extreme'}
    sev_str = SEVERITY_MAP.get(args.severity, 'moderate')
    
    corruptions = [c.strip() for c in args.corruptions.split(',')]
    
    for c in corruptions:
        corr_dir = os.path.join(args.kittic_dir, c, sev_str)
        test_auroc_on_chunk(corr_dir, args.pretrained_path, yaml_labels, yaml_arch, device, c, args.output_dir)

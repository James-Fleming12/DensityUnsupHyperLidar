import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

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
    
    auroc = np.trapz(tpr[::-1], fpr[::-1])
    
    coverage = cum_total / max(cum_total[0], 1)
    precision = np.zeros_like(coverage)
    valid = cum_total > 0
    precision[valid] = cum_correct[valid] / cum_total[valid]
    
    valid_plot = np.where(cum_total > 0)[0]
    if len(valid_plot) > 0:
        first_valid = valid_plot[0]
        coverage = coverage[first_valid:]
        precision = precision[first_valid:]
        
    return auroc, coverage, precision

def extract_clean_features(kitti_dir, model, device, DATA, ARCH):
    print("\n--- Phase 1: Extracting clean features for Oracle K-Means ---")
    parser_obj = Parser(root=kitti_dir,
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
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False, num_workers=8)
    
    MAX_POINTS = 20000 
    class_features = {c: [] for c in range(1, 17)}
    class_counts = {c: 0 for c in range(1, 17)}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if all(c >= MAX_POINTS for c in class_counts.values()):
                break
                
            proj_in = batch_data[0].to(device)
            oracle_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] == 0: continue
            enc_base, _, _ = model.encode(proj_in)
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            if not torch.any(valid_enc_mask): continue
                
            raw_base = F.normalize(enc_base[valid_enc_mask])
            active_oracle = oracle_labels.reshape(-1)[valid_enc_mask]
            
            prototypes = F.normalize(model.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)
            
            for c in range(1, 17):
                if class_counts[c] >= MAX_POINTS: continue
                mask = (active_oracle == c)
                if not torch.any(mask): continue
                
                feats = raw_base[mask].cpu().numpy()
                needed = MAX_POINTS - class_counts[c]
                if feats.shape[0] > needed:
                    indices = np.random.choice(feats.shape[0], needed, replace=False)
                    feats = feats[indices]
                
                class_features[c].append(feats)
                class_counts[c] += feats.shape[0]
            
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"Extracted features from {batch_idx} frames...")

    print("Finished extracting features.")
    
    final_features = {}
    for c in range(1, 17):
        if len(class_features[c]) > 0:
            final_features[c] = np.concatenate(class_features[c], axis=0)
        else:
            final_features[c] = np.zeros((0, raw_base.shape[1]))
            
    return final_features

def build_oracle_subclusters(final_features, K_list=[1, 2, 4, 8, 16]):
    print("\n--- Phase 2: Running K-Means to build Oracle Subclusters ---")
    oracle_subs = {K: {} for K in K_list}
    
    for c in range(1, 17):
        feats = final_features[c]
        n_samples = feats.shape[0]
        
        for K in K_list:
            if n_samples < K:
                if n_samples > 0:
                    centroids = np.vstack([feats.mean(axis=0)] * K)
                else:
                    centroids = np.zeros((K, 2048))
                oracle_subs[K][c] = centroids
                continue
                
            kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
            kmeans.fit(feats)
            
            cents = kmeans.cluster_centers_
            norms = np.linalg.norm(cents, axis=1, keepdims=True)
            cents = np.divide(cents, norms, out=np.zeros_like(cents), where=norms!=0)
            
            oracle_subs[K][c] = cents
            
    print("Finished building Oracles.")
    return oracle_subs

def run_oracle_auroc(corruption_root, model, oracle_subs, K_list, device, corruption_name, DATA, ARCH, output_dir):
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
    
    h_corr_proto = np.zeros(BINS, dtype=np.int64)
    h_incorr_proto = np.zeros(BINS, dtype=np.int64)
    
    h_corr_oracle = {K: np.zeros(BINS, dtype=np.int64) for K in K_list}
    h_incorr_oracle = {K: np.zeros(BINS, dtype=np.int64) for K in K_list}
    
    torch_oracles = {}
    for K in K_list:
        torch_oracles[K] = {c: torch.tensor(oracle_subs[K][c], device=device, dtype=model.classify.weight.dtype) for c in range(1, 17)}
    
    total_points = 0
    total_correct_points = 0
    
    print(f"\n--- Phase 3: Running Oracle AUROC Test on {corruption_name} ---")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            proj_in = batch_data[0].to(device)
            oracle_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] == 0: continue
            enc_base, _, _ = model.encode(proj_in)
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            if not torch.any(valid_enc_mask): continue
                
            raw_base = F.normalize(enc_base[valid_enc_mask])
            active_oracle = oracle_labels.reshape(-1)[valid_enc_mask]
            
            prototypes = F.normalize(model.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)
            
            S_base = raw_base @ prototypes.T
            preds = S_base.argmax(dim=1)
            
            valid_labels_mask = (active_oracle > 0) & (active_oracle < 17) & (preds > 0)
            if not torch.any(valid_labels_mask): continue
                
            filtered_preds = preds[valid_labels_mask]
            filtered_oracle = active_oracle[valid_labels_mask]
            filtered_raw_base = raw_base[valid_labels_mask]
            
            is_correct = (filtered_preds == filtered_oracle).cpu().numpy()
            total_points += len(is_correct)
            total_correct_points += np.sum(is_correct)
            
            # Prototype Similarity
            selected_proto = prototypes[filtered_preds]
            proto_sims = torch.sum(filtered_raw_base * selected_proto, dim=1).cpu().numpy()
            update_hists(is_correct, proto_sims, h_corr_proto, h_incorr_proto)
            
            # Oracle Similarities
            for K in K_list:
                k_sims = torch.zeros(filtered_raw_base.shape[0], device=device)
                for c_id in torch.unique(filtered_preds):
                    c_id_item = c_id.item()
                    c_mask = (filtered_preds == c_id)
                    c_encs = filtered_raw_base[c_mask]
                    
                    c_oracles = torch_oracles[K][c_id_item] 
                    S_subs = c_encs @ c_oracles.T           
                    max_sims, _ = torch.max(S_subs, dim=1)
                    k_sims[c_mask] = max_sims
                    
                update_hists(is_correct, k_sims.cpu().numpy(), h_corr_oracle[K], h_incorr_oracle[K])
            
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"Processed {batch_idx} frames...")
                
    if total_points == 0:
        print("No valid points found to evaluate.")
        return
        
    proto_auroc, cov_proto, prec_proto = compute_metrics_from_hist(h_corr_proto, h_incorr_proto)
    
    print(f"\n=== ORACLE AUROC RESULTS ({corruption_name}) ===")
    print(f"Total Points Evaluated: {total_points:,}")
    print(f"Base Accuracy of Pseudo-Labels: {total_correct_points / total_points:.2%}")
    print(f"AUROC (Prototype): {proto_auroc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cov_proto, prec_proto, label=f'Prototype (AUROC={proto_auroc:.3f})', linewidth=3, color='black', linestyle='--')
    
    for K in K_list:
        k_auroc, k_cov, k_prec = compute_metrics_from_hist(h_corr_oracle[K], h_incorr_oracle[K])
        print(f"AUROC (Oracle K={K}): {k_auroc:.4f}")
        plt.plot(k_cov, k_prec, label=f'Oracle K={K} (AUROC={k_auroc:.3f})', linewidth=2)
        
    print("=====================\n")
    
    plt.title(f'Oracle Gate Quality: Precision vs Coverage ({corruption_name})', fontsize=14)
    plt.xlabel('Coverage (Fraction of points admitted)', fontsize=12)
    plt.ylabel('Precision (Accuracy of admitted points)', fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'oracle_precision_coverage_{corruption_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Oracle Precision-Coverage plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruptions', type=str, default="wet_ground,snow,beam_missing")
    parser.add_argument('--severity', type=int, default=3)
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI')
    parser.add_argument('--kittic_dir', type=str, default='/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth')
    parser.add_argument('--output_dir', type=str, default='logs/oracle_tests')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_labels = 'config/labels/semantic-kitti-all.yaml'
    yaml_arch = 'config/arch/senet-2048p.yml'
    
    DATA = yaml.safe_load(open(yaml_labels, 'r'))
    ARCH = yaml.safe_load(open(yaml_arch, 'r'))
    
    modeldir = os.path.dirname(args.pretrained_path)
    model = AugModel(ARCH, modeldir, 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'), strict=False)
    model = model.to(device)
    model.eval()
    
    final_features = extract_clean_features(args.kitti_dir, model, device, DATA, ARCH)
    
    K_list = [1, 2, 4, 8, 16]
    oracle_subs = build_oracle_subclusters(final_features, K_list=K_list)
    
    SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy', 4: 'extreme'}
    sev_str = SEVERITY_MAP.get(args.severity, 'moderate')
    corruptions = [c.strip() for c in args.corruptions.split(',')]
    
    for c in corruptions:
        corr_dir = os.path.join(args.kittic_dir, c, sev_str)
        run_oracle_auroc(corr_dir, model, oracle_subs, K_list, device, c, DATA, ARCH, args.output_dir)

if __name__ == "__main__":
    main()

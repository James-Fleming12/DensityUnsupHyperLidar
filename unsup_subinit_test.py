import copy
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import types

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from tqdm import tqdm
from faster_mean_shift.mean_shift_cosine_gpu import mean_shift_binary, estimate_bandwidth_binary

from unsup_main import test_hdc_model
from unsup_ugw import get_condition_loaders, save_ablation_dumbbell

MODEL_DIR = "logs"
DATA_DIR = "/mnt/bravo/jmfleming/waymo_skitti"
NUM_CLASSES = 13
HDC_SAVE_PATH = "logs/hdc_sub.pth"

ALL_CONDITIONS = ["sunny", "rain", "night"]
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != "sunny"]

def method_baseline(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
    if len(class_emb_np) == 0: return []
    bw = estimate_bandwidth_binary(class_emb_np, quantile=self.quantile, n_samples=min(500, len(class_emb_np)), bandwidth_multiplier=self.mult)
    cluster_centers = mean_shift_binary(X=class_emb_np, bandwidth=bw, quantile=self.quantile, bandwidth_multiplier=self.mult, dedup_scale=self.dedup)
    if self.subcluster_type == "bipolar": cluster_centers = np.sign(cluster_centers)
    return self._fps_reduce(cluster_centers, num_sub_per_cluster)

def method_increased_samples(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
    if len(class_emb_np) == 0: return []
    # Drop quantile from 0.4 to 0.20, increase n_samples to 2000
    q = 0.20
    bw = estimate_bandwidth_binary(class_emb_np, quantile=q, n_samples=min(2000, len(class_emb_np)), bandwidth_multiplier=self.mult)
    cluster_centers = mean_shift_binary(X=class_emb_np, bandwidth=bw, quantile=q, bandwidth_multiplier=self.mult, dedup_scale=self.dedup)
    if self.subcluster_type == "bipolar": cluster_centers = np.sign(cluster_centers)
    return self._fps_reduce(cluster_centers, num_sub_per_cluster)

def method_iorc(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
    if len(class_emb_np) == 0: return []
    H = torch.tensor(class_emb_np, device=self.device, dtype=torch.float32)
    H = torch.nn.functional.normalize(H, dim=1)
    
    subclusters = []
    for _ in range(num_sub_per_cluster):
        if len(H) == 0: break

        C = torch.sum(H, dim=0)
        norm = torch.norm(C)
        if norm < 1e-6: break
        C = C / norm
        subclusters.append(C.cpu())

        projections = torch.matmul(H, C).unsqueeze(1)
        H_unnorm = H - projections * C.unsqueeze(0)

        norms = torch.norm(H_unnorm, dim=1)
        H = H_unnorm[norms > 1e-6]
        H = torch.nn.functional.normalize(H, dim=1)
        
    return subclusters

def method_adaptive_knn(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
    if len(class_emb_np) == 0: return []
    K = 15
    H = torch.tensor(class_emb_np, device=self.device, dtype=torch.float32)

    if len(H) > 2000:
        H_sample = H[torch.randperm(len(H))[:2000]]
    else:
        H_sample = H
        
    H_sample = torch.nn.functional.normalize(H_sample, dim=1)
    sim_matrix = torch.matmul(H_sample, H_sample.T)

    topk_sims, _ = torch.topk(sim_matrix, min(K+1, len(H_sample)), dim=1)
    topk_dists = 1.0 - topk_sims[:, 1:] 
    sigma_i = topk_dists.mean(dim=1)
    
    centers = H_sample.clone()
    for _ in range(15):
        sim = torch.matmul(H_sample, centers.T)
        dist = 1.0 - sim

        sigma = sigma_i.unsqueeze(1)
        weights = torch.exp(-(dist ** 2) / (2 * (sigma ** 2) + 1e-8))

        new_centers = torch.matmul(weights.T, H_sample)
        new_centers = torch.nn.functional.normalize(new_centers, dim=1)
        
        shift = torch.norm(new_centers - centers, dim=1).max()
        centers = new_centers
        if shift < 1e-4: break

    center_sim = torch.matmul(centers, centers.T)
    keep = torch.ones(len(centers), dtype=torch.bool, device=self.device)
    for i in range(len(centers)):
        if keep[i]:
            close = center_sim[i] > 0.95
            close[i] = False
            keep[close] = False
            
    unique_centers = centers[keep]
    return self._fps_reduce(unique_centers.cpu().numpy(), num_sub_per_cluster)

def _fps_reduce(self, cluster_centers, num_sub_per_cluster):
    num_clusters_found = len(cluster_centers)
    subclusters = []
    if num_clusters_found <= num_sub_per_cluster:
        for center in cluster_centers:
            subclusters.append(torch.tensor(center, device='cpu', dtype=torch.float32))
    else:
        center_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
        fps_indices = self._farthest_point_sample(center_tensor, num_sub_per_cluster)
        for idx in fps_indices.tolist():
            subclusters.append(torch.tensor(cluster_centers[idx], device='cpu', dtype=torch.float32))
    return subclusters

def main():
    print("Loading configs...")
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    except Exception as e:
        print(f"Error opening yaml files. {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seqs = DATA["split"]["train"]
    valid_seqs = DATA["split"]["valid"]

    print("Building per-condition validation loaders...")
    val_loaders = get_condition_loaders(ARCH, DATA, valid_seqs, batch_size=1, shuffle=False, conditions=ALL_CONDITIONS)
    
    ARCH["train"]["workers"] = 0
    print("Building per-condition training loaders...")
    train_loaders = get_condition_loaders(ARCH, DATA, train_seqs, batch_size=1, shuffle=True, conditions=ADVERSE_CONDITIONS)
    
    PRE_DATA = copy.deepcopy(DATA)
    PRE_DATA["weather_filter"] = ["sunny"]
    print("Building sunny dataloader for initialization...")
    sunny_loaders = get_condition_loaders(ARCH, PRE_DATA, train_seqs, batch_size=6, shuffle=True, conditions=["sunny"])
    sunny_loader = sunny_loaders["sunny"]

    init_methods = [
        {"name": "Baseline", "func": method_baseline},
        {"name": "Increased Samples", "func": method_increased_samples},
        {"name": "IORC", "func": method_iorc},
        {"name": "Adaptive KNN", "func": method_adaptive_knn},
    ]

    ablation_histories = []
    
    print("\nEvaluating base model on sunny...")
    model_base = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    model_base.load_state_dict(torch.load(HDC_SAVE_PATH, map_location=device))
    model_base.to(device)
    model_base.eval()
    
    acc_sunny, miou_sunny = test_hdc_model(model_base, val_loaders["sunny"])
    sunny_baseline = {"acc": acc_sunny, "miou": miou_sunny}
    print(f"Baseline Sunny - acc: {acc_sunny:.4f} mIoU: {miou_sunny:.4f}")

    for method_cfg in init_methods:
        history = {
            "name": method_cfg["name"],
            "steps_labels": [],
            "conditions": [],
            "acc_pairs": [],
            "miou_pairs": [],
        }
        
        print(f"\n{'='*60}")
        print(f"Initializing subclusters with method: {method_cfg['name']}")

        model_init = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
        model_init.load_state_dict(torch.load(HDC_SAVE_PATH, map_location=device))
        model_init.to(device)
        model_init.eval()

        model_init._process_single_class = types.MethodType(method_cfg["func"], model_init)
        model_init._fps_reduce = types.MethodType(_fps_reduce, model_init)

        model_init.init_subclusters(sunny_loader)

        saved_subclusters = model_init.subclusters.data.clone()

        for cond in ADVERSE_CONDITIONS:
            if cond not in train_loaders: continue

            print(f"\nCondition: [{cond.upper()}] | Method: {method_cfg['name']}")

            model_cond = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            model_cond.load_state_dict(torch.load(HDC_SAVE_PATH, map_location=device))
            model_cond.subclusters.data = saved_subclusters.clone()
            model_cond.to(device)
            model_cond.eval()

            val_loader_for_cond = val_loaders.get(cond, next(iter(val_loaders.values())))
            
            acc_pre, miou_pre = test_hdc_model(model_cond, val_loader_for_cond)
            print(f"    Pre  - acc: {acc_pre:.4f}  mIoU: {miou_pre:.4f}")

            model_cond.train()
            
            data_iter = iter(train_loaders[cond])
            pbar = tqdm(total=len(train_loaders[cond]), desc=f"    update [{cond}|{method_cfg['name']}]", leave=False)
            
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"\n    [Warning] Skipping bad sample in dataset: {type(e).__name__} - {e}")
                    pbar.update(1)
                    continue

                proj_in = batch[0]
                if proj_in.shape[1] > 0:
                    model_cond.inference_update(proj_in.to(device), learning_rate=0.001, distance_sensitivity=3.0, thresholds=[0.45, 0.80])
                pbar.update(1)
            pbar.close()

            acc_post, miou_post = test_hdc_model(model_cond, val_loader_for_cond)
            print(f"    Post - acc: {acc_post:.4f}  mIoU: {miou_post:.4f}  Δ mIoU: {miou_post - miou_pre:+.4f}")

            history["steps_labels"].append(f"{cond.capitalize()}")
            history["conditions"].append(cond)
            history["acc_pairs"].append((acc_pre, acc_post))
            history["miou_pairs"].append((miou_pre, miou_post))

        ablation_histories.append(history)

    save_ablation_dumbbell(ablation_histories, sunny_baseline=sunny_baseline, file_suffix="_subinit_comparison")
    print("Done! Ablation chart saved.")

if __name__ == "__main__":
    main()

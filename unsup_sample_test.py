import copy
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import types
import heapq

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from tqdm import tqdm

from unsup_main import test_hdc_model
from unsup_ugw import get_condition_loaders, save_ablation_dumbbell

MODEL_DIR = "logs"
DATA_DIR = "/mnt/bravo/jmfleming/waymo_skitti"
NUM_CLASSES = 13
HDC_SAVE_PATH = "logs/hdc_sub.pth"

ALL_CONDITIONS = ["sunny", "rain", "night"]
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != "sunny"]

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

def adaptive_knn_process(self, class_emb_np, num_sub_per_cluster):
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


# --- SAMPLING METHODS ---

def collect_baseline(self, dataloader, class_id, max_samples=8000):
    MAX_SAMPLES = max_samples * 2
    class_embeddings = []
    batch_indices = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            proj_in = batch[0].to(self.device)
            proj_labels = batch[2].to(self.device).flatten()
            
            valid_mask = proj_labels >= 0
            if not valid_mask.any(): continue
            proj_labels = proj_labels[valid_mask]
            enc, _, _ = self.encode(proj_in)
            enc = enc[valid_mask]
            
            class_mask = proj_labels == class_id
            if torch.any(class_mask):
                class_enc = enc[class_mask].cpu().half()
                class_embeddings.append(class_enc)
                batch_indices.extend([batch_idx] * class_enc.shape[0])
                total_samples += class_enc.shape[0]
            
            if total_samples >= MAX_SAMPLES: break
            
    if not class_embeddings: return np.array([])
    
    class_emb_cpu = torch.cat(class_embeddings, dim=0)
    batch_indices = torch.as_tensor(batch_indices)
    
    if len(class_emb_cpu) > MAX_SAMPLES:
        indices = torch.randperm(len(class_emb_cpu))[:MAX_SAMPLES]
        class_emb_cpu = class_emb_cpu[indices]
        batch_indices = batch_indices[indices]
        
    if len(class_emb_cpu) > max_samples:
        indices = self._stratified_sample(batch_indices, max_samples)
        class_emb_cpu = class_emb_cpu[indices]
        
    return class_emb_cpu.numpy()


def collect_reservoir(self, dataloader, class_id, max_samples=8000):
    global_sum = None
    total_samples = 0
    MAX_STREAM = max_samples * 2
    with torch.no_grad():
        for batch in dataloader:
            proj_in = batch[0].to(self.device)
            proj_labels = batch[2].to(self.device).flatten()
            valid_mask = proj_labels >= 0
            if not valid_mask.any(): continue
            proj_labels = proj_labels[valid_mask]
            enc, _, _ = self.encode(proj_in)
            enc = enc[valid_mask]
            class_mask = proj_labels == class_id
            if torch.any(class_mask):
                class_enc = enc[class_mask]
                if global_sum is None: global_sum = class_enc.sum(dim=0)
                else: global_sum += class_enc.sum(dim=0)
                
                total_samples += class_enc.shape[0]
            if total_samples >= MAX_STREAM: break
                
    if global_sum is None: return np.array([])
    Pglobal = torch.nn.functional.normalize(global_sum, dim=0)
    
    reservoir = [] # (key, id, emb)
    item_id = 0
    with torch.no_grad():
        for batch in dataloader:
            proj_in = batch[0].to(self.device)
            proj_labels = batch[2].to(self.device).flatten()
            valid_mask = proj_labels >= 0
            if not valid_mask.any(): continue
            proj_labels = proj_labels[valid_mask]
            enc, _, _ = self.encode(proj_in)
            enc = enc[valid_mask]
            class_mask = proj_labels == class_id
            if not torch.any(class_mask): continue
            
            class_enc = enc[class_mask]
            sims = torch.matmul(torch.nn.functional.normalize(class_enc, dim=1), Pglobal)
            dists = 1.0 - sims
            weights = dists + 1e-4 
            
            u = torch.rand_like(weights)
            keys = u ** (1.0 / weights)
            
            for i in range(len(keys)):
                key = keys[i].item()
                emb = class_enc[i].cpu().half()
                if len(reservoir) < max_samples:
                    heapq.heappush(reservoir, (key, item_id, emb))
                else:
                    if key > reservoir[0][0]:
                        heapq.heappushpop(reservoir, (key, item_id, emb))
                item_id += 1
            
            total_samples += class_enc.shape[0]
            if total_samples >= MAX_STREAM: break
                
    final_embs = [x[2] for x in reservoir]
    return torch.stack(final_embs).numpy()


def collect_sieve(self, dataloader, class_id, max_samples=8000):
    buffer_embs = []
    buffer_sum = None
    tau = 0.5
    decay_rate = 0.9999
    total_samples = 0
    MAX_STREAM = max_samples * 2
    
    with torch.no_grad():
        for batch in dataloader:
            proj_in = batch[0].to(self.device)
            proj_labels = batch[2].to(self.device).flatten()
            valid_mask = proj_labels >= 0
            if not valid_mask.any(): continue
            proj_labels = proj_labels[valid_mask]
            enc, _, _ = self.encode(proj_in)
            enc = enc[valid_mask]
            class_mask = proj_labels == class_id
            if not torch.any(class_mask): continue
            
            class_enc = enc[class_mask]
            
            for hi in class_enc:
                hi_norm = torch.nn.functional.normalize(hi, dim=0)
                if buffer_sum is None:
                    buffer_embs.append(hi.cpu().half())
                    buffer_sum = hi.clone()
                    continue
                    
                bs_norm = torch.nn.functional.normalize(buffer_sum, dim=0)
                sim = torch.dot(hi_norm, bs_norm)
                delta = 1.0 - sim.item()
                
                if delta > tau:
                    buffer_embs.append(hi.cpu().half())
                    buffer_sum += hi
                    
                tau *= decay_rate
                
            total_samples += class_enc.shape[0]
            if total_samples >= MAX_STREAM: break
                
            if len(buffer_embs) > max_samples + 500:
                S_tensor = torch.stack(buffer_embs).to(self.device).float()
                S_norm = torch.nn.functional.normalize(S_tensor, dim=1)
                sims = torch.matmul(S_norm, S_norm.T)
                sims.fill_diagonal_(-1)
                
                num_to_drop = len(buffer_embs) - max_samples
                K = min(15, len(buffer_embs)-1)
                topk_sims, _ = torch.topk(sims, K, dim=1)
                avg_sims = topk_sims.mean(dim=1)
                
                _, drop_indices = torch.topk(avg_sims, num_to_drop)
                drop_indices = drop_indices.cpu().tolist()
                drop_indices.sort(reverse=True)
                
                for idx in drop_indices:
                    dropped_emb = buffer_embs.pop(idx)
                    buffer_sum -= dropped_emb.to(self.device).float()
                    
    # Final prune
    if len(buffer_embs) > max_samples:
        S_tensor = torch.stack(buffer_embs).to(self.device).float()
        S_norm = torch.nn.functional.normalize(S_tensor, dim=1)
        sims = torch.matmul(S_norm, S_norm.T)
        sims.fill_diagonal_(-1)
        K = min(15, len(buffer_embs)-1)
        topk_sims, _ = torch.topk(sims, K, dim=1)
        avg_sims = topk_sims.mean(dim=1)
        num_to_drop = len(buffer_embs) - max_samples
        _, drop_indices = torch.topk(avg_sims, num_to_drop)
        drop_indices = drop_indices.cpu().tolist()
        drop_indices.sort(reverse=True)
        for idx in drop_indices:
            buffer_embs.pop(idx)
            
    if not buffer_embs: return np.array([])
    return torch.stack(buffer_embs).numpy()


def collect_minibatch_coreset(self, dataloader, class_id, max_samples=8000):
    S = []
    total_samples = 0
    MAX_STREAM = max_samples * 2
    
    with torch.no_grad():
        for batch in dataloader:
            proj_in = batch[0].to(self.device)
            proj_labels = batch[2].to(self.device).flatten()
            valid_mask = proj_labels >= 0
            if not valid_mask.any(): continue
            proj_labels = proj_labels[valid_mask]
            enc, _, _ = self.encode(proj_in)
            enc = enc[valid_mask]
            class_mask = proj_labels == class_id
            if not torch.any(class_mask): continue
            
            class_enc = enc[class_mask]
            for hi in class_enc:
                S.append(hi.cpu().half())
                
            total_samples += class_enc.shape[0]
            if total_samples >= MAX_STREAM: break
            
            if len(S) > max_samples + 1000:
                S_tensor = torch.stack(S).to(self.device).float()
                S_norm = torch.nn.functional.normalize(S_tensor, dim=1)
                sims = torch.matmul(S_norm, S_norm.T)
                sims.fill_diagonal_(-1)
                
                K = min(15, len(S)-1)
                topk_sims, _ = torch.topk(sims, K, dim=1)
                avg_sims = topk_sims.mean(dim=1)
                
                num_to_drop = len(S) - max_samples
                _, drop_indices = torch.topk(avg_sims, num_to_drop)
                drop_indices = drop_indices.cpu().tolist()
                drop_indices.sort(reverse=True)
                for idx in drop_indices:
                    S.pop(idx)
                    
    if len(S) > max_samples:
        S_tensor = torch.stack(S).to(self.device).float()
        S_norm = torch.nn.functional.normalize(S_tensor, dim=1)
        sims = torch.matmul(S_norm, S_norm.T)
        sims.fill_diagonal_(-1)
        K = min(15, len(S)-1)
        topk_sims, _ = torch.topk(sims, K, dim=1)
        avg_sims = topk_sims.mean(dim=1)
        num_to_drop = len(S) - max_samples
        _, drop_indices = torch.topk(avg_sims, num_to_drop)
        drop_indices = drop_indices.cpu().tolist()
        drop_indices.sort(reverse=True)
        for idx in drop_indices:
            S.pop(idx)

    if not S: return np.array([])
    return torch.stack(S).numpy()


def collect_dpp(self, dataloader, class_id, max_samples=8000):
    buffer_embs = []
    buffer_sum = None
    total_samples = 0
    MAX_STREAM = max_samples * 2
    
    with torch.no_grad():
        for batch in dataloader:
            proj_in = batch[0].to(self.device)
            proj_labels = batch[2].to(self.device).flatten()
            valid_mask = proj_labels >= 0
            if not valid_mask.any(): continue
            proj_labels = proj_labels[valid_mask]
            enc, _, _ = self.encode(proj_in)
            enc = enc[valid_mask]
            class_mask = proj_labels == class_id
            if not torch.any(class_mask): continue
            
            class_enc = enc[class_mask]
            
            for hi in class_enc:
                if len(buffer_embs) == 0:
                    buffer_embs.append(hi.cpu().half())
                    buffer_sum = hi.clone()
                    continue
                    
                hi_norm = torch.nn.functional.normalize(hi, dim=0)
                bs_norm = torch.nn.functional.normalize(buffer_sum, dim=0)
                
                proj = torch.dot(hi_norm, bs_norm) * bs_norm
                residual = hi_norm - proj
                res_mag = torch.norm(residual).item()
                
                prob = min(res_mag * 2.0, 1.0) 
                if np.random.rand() < prob:
                    buffer_embs.append(hi.cpu().half())
                    buffer_sum += hi
                    
            total_samples += class_enc.shape[0]
            if total_samples >= MAX_STREAM: break
                    
    if not buffer_embs: return np.array([])
    
    S_tensor = torch.stack(buffer_embs)
    if len(S_tensor) > max_samples:
        indices = torch.randperm(len(S_tensor))[:max_samples]
        S_tensor = S_tensor[indices]
        
    return S_tensor.numpy()

# --- WRAPPER FOR INIT ---

def make_custom_init(collection_func):
    def custom_init(self, dataloader, max_samples_per_class=8000):
        self.eval()
        num_sub_per_cluster = self.num_subclusters
        all_subcluster_centers = []
        all_subcluster_classes = []

        for class_id in range(self.num_classes):
            print(f"Processing class {class_id}...")
            
            class_emb_np = collection_func(self, dataloader, class_id, max_samples_per_class)
            
            if class_emb_np is None or len(class_emb_np) == 0:
                print(f"  No data for class {class_id}, skipping")
                continue
                
            print(f"  Using {len(class_emb_np)} samples for clustering")
            subclusters_for_class = adaptive_knn_process(self, class_emb_np, num_sub_per_cluster)
            
            all_subcluster_centers.extend(subclusters_for_class)
            all_subcluster_classes.extend([class_id] * len(subclusters_for_class))
            
            self._clear_memory()
            
        self._load_subclusters(all_subcluster_centers, all_subcluster_classes)
        print("Subcluster initialization complete")
    return custom_init


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

    sample_methods = [
        {"name": "Baseline (Diverse)", "func": make_custom_init(collect_baseline)},
        {"name": "A-Res Coreset", "func": make_custom_init(collect_reservoir)},
        {"name": "Sieve-Streaming", "func": make_custom_init(collect_sieve)},
        {"name": "Mini-Batch Coreset", "func": make_custom_init(collect_minibatch_coreset)},
        {"name": "HDC DPP", "func": make_custom_init(collect_dpp)},
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

    for method_cfg in sample_methods:
        history = {
            "name": method_cfg["name"],
            "steps_labels": [],
            "conditions": [],
            "acc_pairs": [],
            "miou_pairs": [],
        }
        
        print(f"\n{'='*60}")
        print(f"Initializing subclusters with sampling method: {method_cfg['name']}")
        
        model_init = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
        model_init.load_state_dict(torch.load(HDC_SAVE_PATH, map_location=device))
        model_init.to(device)
        model_init.eval()
        
        # Override the subclusters init logic to use our sampler + adaptive KNN processing
        model_init.init_subclusters = types.MethodType(method_cfg["func"], model_init)
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
                    model_cond.inference_update(
                        proj_in.to(device),
                        learning_rate=0.001,
                        distance_sensitivity=3.0,
                        thresholds=[0.45, 0.80]
                    )
                pbar.update(1)
            pbar.close()

            acc_post, miou_post = test_hdc_model(model_cond, val_loader_for_cond)
            print(f"    Post - acc: {acc_post:.4f}  mIoU: {miou_post:.4f}  Δ mIoU: {miou_post - miou_pre:+.4f}")

            history["steps_labels"].append(f"{cond.capitalize()}")
            history["conditions"].append(cond)
            history["acc_pairs"].append((acc_pre, acc_post))
            history["miou_pairs"].append((miou_pre, miou_post))

        ablation_histories.append(history)

    save_ablation_dumbbell(ablation_histories, sunny_baseline=sunny_baseline, file_suffix="_sample_comparison")
    print("Done! Ablation chart saved.")

if __name__ == "__main__":
    main()

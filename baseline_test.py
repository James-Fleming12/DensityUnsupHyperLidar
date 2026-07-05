import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import yaml
import importlib
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel

# Dynamically import LiDARCorruptionWrapper
unsup_kitti_c = importlib.import_module("unsup_kitti-c")
LiDARCorruptionWrapper = unsup_kitti_c.LiDARCorruptionWrapper

# Configuration
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
NUM_CLASSES = 17
CONFIG_PATH = "config/arch/senet-2048p.yml"
LABELS_PATH = "config/labels/semantic-kitti-all.yaml"
MODEL_DIR = "logs/kitti_pretrain"
HDC_SUB_PATH = os.path.join(MODEL_DIR, "hdc_sub.pth")
SAVE_DIR = "logs/diagnostics"

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    with open(CONFIG_PATH, 'r') as f:
        ARCH = yaml.safe_load(f)
        
    with open(LABELS_PATH, 'r') as f:
        DATA = yaml.safe_load(f)
        
    train_seqs = DATA["split"]["train"][:4]
    
    # Setup Dataset
    print("Building SemanticKITTI clean baseline parser...")
    baseline_parser = Parser(
        root=DATA_DIR,
        train_sequences=train_seqs,
        valid_sequences=DATA["split"]["valid"],
        test_sequences=None,
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=ARCH["train"]["workers"],
        gt=True,
        shuffle_train=False
    )
    
    raw_train_dataset = baseline_parser.get_train_set().dataset
    # Subsample for faster testing across conditions
    raw_train_dataset.scan_files = raw_train_dataset.scan_files[::4]
    raw_train_dataset.label_files = raw_train_dataset.label_files[::4]
    
    print("Loading Original DensityModel Baseline...")
    model_base = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    
    loaded_obj = torch.load(HDC_SUB_PATH, map_location=device, weights_only=False)
    state_dict = loaded_obj.state_dict() if isinstance(loaded_obj, torch.nn.Module) else loaded_obj
    
    if "subclusters" in state_dict:
        new_size = state_dict["subclusters"].shape[0]
        if model_base.subclusters.shape[0] != new_size:
            model_base.subclusters = torch.nn.Parameter(torch.zeros(new_size, model_base.hd_dim, device=device))
            
    model_base.load_state_dict(state_dict, strict=False)
    model_base.to(device)
    
    # Run across different conditions and adaptation modes
    conditions = ["snow", "fog"]
    modes = [False, True] # False = Frozen Day-0 evaluation, True = Actively adapting
    
    for cond in conditions:
        for adapt_active in modes:
            
            mode_str = "Adapting" if adapt_active else "Frozen"
            print(f"\n{'='*60}")
            print(f"Running Diagnostic on {cond.upper()} (Severity 3) - Mode: {mode_str}")
            
            target_dataset = LiDARCorruptionWrapper(raw_train_dataset, corruption_type=cond, severity=3)
            data_loader = DataLoader(target_dataset, batch_size=ARCH["train"]["batch_size"], shuffle=False, num_workers=ARCH["train"]["workers"], drop_last=False)
            
            # Initialize trackers
            T1_stats = {rz: {dz: {"correct": [0]*NUM_CLASSES, "total": [0]*NUM_CLASSES} for dz in range(3)} for rz in range(3)}
            T2_confusion_hists = {} 
            T3_margin_hists = {rz: {"correct": [0]*20, "incorrect": [0]*20} for rz in range(3)}
            T4_prototype_snapshots = []
            T5_subcluster_hist = {c: [] for c in range(NUM_CLASSES)}
            T6_drift = {c: {"cos_to_origin": [], "delta_consistency": []} for c in range(NUM_CLASSES)}
            T7_calibration = {rz: {"conf_bins": [0]*10, "correct_bins": [0]*10, "total_bins": [0]*10} for rz in range(3)}
            
            source_prototypes = model_base.classify.weight.detach().clone()
            _prev_protos = source_prototypes.clone()
            _prev_deltas = {c: None for c in range(NUM_CLASSES)}
            
            model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            if model.subclusters.shape[0] != model_base.subclusters.shape[0]:
                model.subclusters = torch.nn.Parameter(torch.zeros(model_base.subclusters.shape[0], model.hd_dim, device=device))
            model.load_state_dict(model_base.state_dict(), strict=False)
            model.to(device)
            model.train() if adapt_active else model.eval()
            
            for step, batch in enumerate(tqdm(data_loader, desc=f"Diagnostics [{cond} | {mode_str}]")):
                proj_in = batch[0].to(device)
                oracle_labels = batch[2].to(device) if len(batch) > 2 else None
                proj_xyz = batch[10].to(device) if len(batch) > 10 else None
                
                if proj_in.shape[1] == 0 or oracle_labels is None or proj_xyz is None:
                    continue
                    
                with torch.no_grad():
                    enc_base, _, _ = model.encode(proj_in)
                    valid_mask_3d = (enc_base.abs().sum(dim=1) > 0)
                    
                    valid_mask_2d = (proj_in.abs().sum(dim=1) > 0).float()
                    local_density_2d = F.avg_pool2d(valid_mask_2d.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
                    active_density = local_density_2d.view(-1)[valid_mask_3d.view(-1)]
                    
                    raw_base = F.normalize(enc_base[valid_mask_3d])
                    active_labels = oracle_labels.view(-1)[valid_mask_3d.view(-1)]
                    xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                    active_xyz = xyz_flat[valid_mask_3d.view(-1)]
                    
                    if raw_base.shape[0] == 0:
                        continue
                        
                    prototypes = F.normalize(model.classify.weight)
                    raw_base = raw_base.to(prototypes.dtype)
                    
                    S = raw_base @ prototypes.T
                    top2_sims, top2_preds = torch.topk(S, k=2, dim=1)
                    preds = top2_preds[:, 0]
                    sims = top2_sims[:, 0]
                    margins = top2_sims[:, 0] - top2_sims[:, 1]
                    
                    radial_dist = torch.norm(active_xyz, dim=1)
                    rad_zones = torch.zeros_like(radial_dist, dtype=torch.long)
                    rad_zones[(radial_dist >= 15) & (radial_dist < 30)] = 1
                    rad_zones[radial_dist >= 30] = 2
                    
                    den_zones = torch.zeros_like(active_density, dtype=torch.long)
                    den_zones[(active_density >= 0.33) & (active_density < 0.66)] = 1
                    den_zones[active_density >= 0.66] = 2
                    
                    if step % 50 == 0:
                        for c_id in range(NUM_CLASSES):
                            c_mask = (active_labels == c_id)
                            if c_mask.sum() > 0:
                                c_encs = raw_base[c_mask]
                                # Updated distance_sensitivity to 3.0 to match the actual update
                                _, sub_idx = model.get_max_subcluster_similarity(c_encs, c_id, distance_sensitivity=3.0)
                                rel_idx = sub_idx % model.num_subclusters
                                hist = torch.bincount(rel_idx, minlength=model.num_subclusters)
                                T5_subcluster_hist[c_id].append(hist.cpu().numpy().tolist())
                                
                    correct_mask = (preds == active_labels)
                    
                    for rz in range(3):
                        rz_mask = (rad_zones == rz)
                        if not torch.any(rz_mask): continue
                        
                        z_margins_correct = margins[rz_mask & correct_mask]
                        z_margins_incorrect = margins[rz_mask & ~correct_mask]
                        
                        if z_margins_correct.numel() > 0:
                            bin_c = (z_margins_correct * 20).clamp(0, 19).long()
                            for b in bin_c.tolist(): T3_margin_hists[rz]["correct"][b] += 1
                        if z_margins_incorrect.numel() > 0:
                            bin_i = (z_margins_incorrect * 20).clamp(0, 19).long()
                            for b in bin_i.tolist(): T3_margin_hists[rz]["incorrect"][b] += 1
                        
                        z_confs = sims[rz_mask]
                        z_corr = correct_mask[rz_mask]
                        bin_idx = (z_confs * 10).clamp(0, 9).long()
                        for b in bin_idx.tolist(): T7_calibration[rz]["total_bins"][b] += 1
                        for b, corr in zip(bin_idx.tolist(), z_corr.tolist()): T7_calibration[rz]["correct_bins"][b] += int(corr)
                        
                        for dz in range(3):
                            dz_mask = rz_mask & (den_zones == dz)
                            if not torch.any(dz_mask): continue
                            
                            z_correct = active_labels[dz_mask & correct_mask]
                            z_total = active_labels[dz_mask]
                            
                            for c in z_correct.tolist(): T1_stats[rz][dz]["correct"][c] += 1
                            for c in z_total.tolist(): T1_stats[rz][dz]["total"][c] += 1
                            
                    incorrect_mask = ~correct_mask
                    if torch.any(incorrect_mask):
                        inc_preds = preds[incorrect_mask]
                        inc_trues = active_labels[incorrect_mask]
                        inc_confs = sims[incorrect_mask]
                        
                        bin_idxs = (inc_confs * 20).clamp(0, 19).long()
                        
                        for p, t, b in zip(inc_preds.tolist(), inc_trues.tolist(), bin_idxs.tolist()):
                            pair = f"{p}_{t}"
                            if pair not in T2_confusion_hists:
                                T2_confusion_hists[pair] = [0]*20
                            T2_confusion_hists[pair][b] += 1
                            
                # Apply the unsupervised update if adapting
                if adapt_active:
                    model.inference_update(
                        proj_in,
                        learning_rate=0.001,
                        distance_sensitivity=3.0,
                        thresholds=[0.45, 0.80]
                    )
                
                # Test 4 & 6: Prototype tracking
                if step % 50 == 0:
                    curr_proto = F.normalize(model.classify.weight.detach())
                    T4_prototype_snapshots.append((curr_proto @ curr_proto.T).cpu().numpy().tolist())
                    
                    for c_id in range(NUM_CLASSES):
                        cos_orig = F.cosine_similarity(curr_proto[c_id:c_id+1], F.normalize(source_prototypes[c_id:c_id+1])).item()
                        T6_drift[c_id]["cos_to_origin"].append(cos_orig)
                        
                        delta = curr_proto[c_id] - _prev_protos[c_id]
                        if _prev_deltas[c_id] is not None:
                            consistency = F.cosine_similarity(delta.unsqueeze(0), _prev_deltas[c_id].unsqueeze(0)).item()
                            T6_drift[c_id]["delta_consistency"].append(consistency)
                        _prev_deltas[c_id] = delta
                        
                    _prev_protos = curr_proto.clone()
                    
            stats = {
                "T1_radial_density_stats": T1_stats,
                "T2_confusion_hists": T2_confusion_hists,
                "T3_margin_hists": T3_margin_hists,
                "T4_prototype_snapshots": T4_prototype_snapshots,
                "T5_subcluster_hist": T5_subcluster_hist,
                "T6_drift": T6_drift,
                "T7_calibration": T7_calibration
            }
            
            out_path = os.path.join(SAVE_DIR, f"baseline_diagnostics_{cond}_{mode_str.lower()}.json")
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=4)
                
            print(f"Diagnostics completed! Data saved to {out_path}")

if __name__ == "__main__":
    run_diagnostics()

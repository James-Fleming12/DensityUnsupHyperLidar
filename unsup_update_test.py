import argparse
import copy
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import json
import time
import importlib

from dataset.kitti.parser import Parser
from modules.aug_model import AugModel
from torch.utils.data import DataLoader

from tqdm import tqdm
from unsup_main import test_hdc_model

# Dynamically import LiDARCorruptionWrapper
unsup_kitti_c = importlib.import_module("unsup_kitti-c")
LiDARCorruptionWrapper = unsup_kitti_c.LiDARCorruptionWrapper

MODEL_DIR = "logs/kitti_pretrain"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
NUM_CLASSES = 17
CONFIG_PATH = "config/arch/senet-2048p.yml"
LABELS_PATH = "config/labels/semantic-kitti-all.yaml"
HDC_SUB_PATH = os.path.join(MODEL_DIR, "hdc_sub.pth")
SAVE_DIR = "logs/diagnostics"

ALL_CONDITIONS = ["snow", "fog"]

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Update Methods")
    parser.add_argument('--dry-run', action='store_true', help='Test methods with a single sample')
    args = parser.parse_args()
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        ARCH = yaml.safe_load(open(CONFIG_PATH, 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open(LABELS_PATH, 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seqs = DATA["split"]["train"][:1]
    valid_seqs = DATA["split"]["valid"]

    print("Building SemanticKITTI clean baseline parser...")
    baseline_parser = Parser(
        root=DATA_DIR,
        train_sequences=train_seqs,
        valid_sequences=valid_seqs,
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
    # Subsample training data by taking every 4th frame
    raw_train_dataset.scan_files = raw_train_dataset.scan_files[::4]
    raw_train_dataset.label_files = raw_train_dataset.label_files[::4]

    valid_dataset = baseline_parser.validloader.dataset
    # Subsample validation data by taking every 10th frame
    valid_dataset.scan_files = valid_dataset.scan_files[::10]
    valid_dataset.label_files = valid_dataset.label_files[::10]
    
    val_loaders = {"sunny": DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])}

    update_methods = [
        {"name": "Frozen Baseline", "method": None, "is_active": False},
        {"name": "Soft Multi-View Consensus (Exp A)", "method": "inference_update_soft_consensus"},
        {"name": "Soft Multi-View Consensus + Density Weighting (Exp B)", "method": "inference_update_soft_dcsp"},
        {"name": "Multi-Scale Spatial Bundling (MSSB)", "method": "inference_update_mssb"},
        {"name": "Orthogonalized Prototype Pull (OPP)", "method": "inference_update_opp"},
        {"name": "Coverage-Weighted Subcluster Allocation (CWSA)", "method": "inference_update_cwsa"},
        {"name": "Low-Threshold Consensus Gating (LTCG)", "method": "inference_update_ltcg"}
    ]

    model_base = AugModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    
    loaded_obj = torch.load(HDC_SUB_PATH, map_location=device, weights_only=False)
    state_dict = loaded_obj.state_dict() if isinstance(loaded_obj, torch.nn.Module) else loaded_obj
    
    if "subclusters" in state_dict:
        new_size = state_dict["subclusters"].shape[0]
        if model_base.subclusters.shape[0] != new_size:
            model_base.subclusters = torch.nn.Parameter(torch.zeros(new_size, model_base.hd_dim, device=device))
            
    model_base.load_state_dict(state_dict, strict=False)
    model_base.to(device)
    
    print("\nEvaluating baseline on sunny...")
    acc_sunny, miou_sunny = test_hdc_model(model_base, val_loaders["sunny"])
    print(f"Baseline Sunny - acc: {acc_sunny:.4f} mIoU: {miou_sunny:.4f}")
    
    history_logs = []
    condition_baselines = {}
    
    for cfg in update_methods:
        history = {
            "name": cfg["name"],
            "conditions": [],
            "acc_pairs": [],
            "miou_pairs": []
        }
        
        for cond in ALL_CONDITIONS:
            print(f"\n{'='*60}")
            print(f"Condition: [{cond.upper()}] | Method: {cfg['name']}")

            model = AugModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            if model.subclusters.shape[0] != model_base.subclusters.shape[0]:
                model.subclusters = torch.nn.Parameter(torch.zeros(model_base.subclusters.shape[0], model.hd_dim, device=device))
            model.load_state_dict(model_base.state_dict(), strict=False)
            model.to(device)
            
            is_active = cfg.get("is_active", True)
            model.eval()  # always eval — F.dropout2d applies regardless via training=True default,
                          # and this keeps BatchNorm on frozen running stats throughout adaptation

            # Baseline for pairs
            val_target_dataset = LiDARCorruptionWrapper(valid_dataset, corruption_type=cond, severity=3)
            val_loader = DataLoader(val_target_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            if cond not in condition_baselines:
                eval_loader = [next(iter(val_loader))] if args.dry_run else val_loader
                b_acc, b_miou = test_hdc_model(model_base, eval_loader)
                condition_baselines[cond] = (b_acc, b_miou)
            
            b_acc, b_miou = condition_baselines[cond]

            # Adapt on train set
            if is_active:
                target_dataset = LiDARCorruptionWrapper(raw_train_dataset, corruption_type=cond, severity=3)
                train_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
                
                for batch in tqdm(train_loader, desc=f"Adapting [{cond}]"):
                    proj_in = batch[0].to(device)
                    proj_xyz = batch[10].to(device) if len(batch) > 10 else None
                    if proj_in.shape[1] == 0:
                        continue
                        
                    method_func = getattr(model, cfg["method"])
                    method_func(
                        proj_in, 
                        proj_xyz=proj_xyz, 
                        thresholds=[0.40, 0.65], 
                        learning_rate=0.001
                    )
                    if args.dry_run:
                        break
            
            # Evaluate on valid set
            print(f"Evaluating {cfg['name']} on {cond}...")
            
            if args.dry_run:
                val_loader = [next(iter(val_loader))]
                
            acc, miou = test_hdc_model(model, val_loader)
            print(f"Result - acc: {acc:.4f} mIoU: {miou:.4f}")
            
            history["conditions"].append(cond)
            history["acc_pairs"].append([b_acc, acc])
            history["miou_pairs"].append([b_miou, miou])
            
        history_logs.append(history)
        
        with open(os.path.join(SAVE_DIR, "hypothesis_testing_results.json"), "w") as f:
            json.dump(history_logs, f, indent=4)
            
    # Draw graph
    from unsup_ugw import save_ablation_dumbbell
    save_ablation_dumbbell(history_logs, sunny_baseline={"acc": acc_sunny, "miou": miou_sunny}, file_suffix="_hypotheses")

if __name__ == "__main__":
    main()

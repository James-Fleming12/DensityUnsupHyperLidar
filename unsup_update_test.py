import copy
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import copy

from dataset.kitti.parser import Parser
from modules.HDC_utils import Model, DensityModel
from modules.active_model import ActiveModel

from tqdm import tqdm

from unsup_main import test_hdc_model
from unsup_ugw import get_condition_loaders, save_ablation_dumbbell, save_multi_step_dumbbell_ug

MODEL_DIR = "logs"
DATA_DIR = "/mnt/bravo/jmfleming/waymo_skitti"
NUM_CLASSES = 13
HDC_SUB_PATH = "logs/hdc_sub.pth"

ALL_CONDITIONS = ["sunny", "rain", "night"]
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != "sunny"]

def main():
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seqs = DATA["split"]["train"]
    valid_seqs = DATA["split"]["valid"]

    print("Building per-condition validation loaders...")
    val_loaders = get_condition_loaders(
        ARCH, DATA, valid_seqs,
        batch_size=1, shuffle=False,
        conditions=ALL_CONDITIONS)

    if not val_loaders:
        raise RuntimeError("No validation frames found for any condition.")

    ARCH["train"]["workers"] = 0 # Set to 0 to prevent worker crashes on bad dataset samples
    print("Building per-condition training loaders (for adaptation)...")
    train_loaders = get_condition_loaders(
        ARCH, DATA, train_seqs,
        batch_size=1, shuffle=True,
        conditions=ADVERSE_CONDITIONS)

    if not train_loaders:
        print("No adverse condition frames found - skipping.")
        return

    update_methods = [
        {"name": "Outlier Oracle Anchor (ADA)", "method": "inference_update_ooa", "is_active": True},
        {"name": "Hypervector Bundling (TTAug)", "method": "inference_update_ttaug", "is_active": False},
        {"name": "Graph-Laplacian Label Propagation", "method": "inference_update_gplp", "is_active": False},
    ]

    ablation_histories = []
    
    # Store sunny baseline performance
    model_base = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    model_base.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device))
    model_base.to(device)
    
    print("\nEvaluating baseline on sunny...")
    acc_sunny, miou_sunny = test_hdc_model(model_base, val_loaders["sunny"])
    sunny_baseline = {"acc": acc_sunny, "miou": miou_sunny}
    print(f"Baseline Sunny - acc: {acc_sunny:.4f} mIoU: {miou_sunny:.4f}")

    for cfg in update_methods:
        history = {
            "name": cfg["name"],
            "steps_labels": [],
            "conditions": [],
            "acc_pairs": [],
            "miou_pairs": [],
        }

        for cond in ADVERSE_CONDITIONS:
            if cond not in train_loaders:
                continue

            print(f"\n{'='*60}")
            print(f"Condition: [{cond.upper()}] | Method: {cfg['name']}")

            val_loader_for_cond = val_loaders.get(cond, next(iter(val_loaders.values())))

            if cfg.get("is_active", False):
                model = ActiveModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            else:
                model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            model.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device))
            model.to(device)

            acc_pre, miou_pre = test_hdc_model(model, val_loader_for_cond)
            print(f"    Pre  - acc: {acc_pre:.4f}  mIoU: {miou_pre:.4f}")

            model.train()
            
            update_fn = getattr(model, cfg["method"])
            
            data_iter = iter(train_loaders[cond])
            pbar = tqdm(total=len(train_loaders[cond]), desc=f"    update [{cond}|{cfg['name']}]", leave=False)
            
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
                oracle_labels = batch[2] if len(batch) > 2 else None
                proj_xyz = batch[10] if len(batch) > 10 else None
                
                kwargs = {}
                if cfg["method"] != "inference_update":
                    kwargs["oracle_labels"] = oracle_labels.to(device) if oracle_labels is not None else None
                if cfg["method"] in ["inference_update_gplp", "inference_update_ooa", "inference_update_ttaug"]:
                    kwargs["proj_xyz"] = proj_xyz.to(device) if proj_xyz is not None else None
                
                if proj_in.shape[1] > 0:
                    update_fn(
                        proj_in.to(device),
                        learning_rate=0.001,
                        distance_sensitivity=3.0,
                        thresholds=[0.45, 0.80], # Matching Baseline from unsup_ugw.py
                        **kwargs
                    )
                pbar.update(1)
            pbar.close()

            acc_post, miou_post = test_hdc_model(model, val_loader_for_cond)
            print(f"    Post - acc: {acc_post:.4f}  mIoU: {miou_post:.4f}  Δ mIoU: {miou_post - miou_pre:+.4f}")

            history["steps_labels"].append(f"{cond.capitalize()}")
            history["conditions"].append(cond)
            history["acc_pairs"].append((acc_pre, acc_post))
            history["miou_pairs"].append((miou_pre, miou_post))

        ablation_histories.append(history)

    save_ablation_dumbbell(ablation_histories, sunny_baseline=sunny_baseline, file_suffix="_methods_comparison")

if __name__ == "__main__":
    main()

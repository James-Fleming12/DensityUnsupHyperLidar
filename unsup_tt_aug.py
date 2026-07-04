import argparse
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import json
import time

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from modules.aug_model import AugModel, AugTrainer
from torch.utils.data import DataLoader

from tqdm import tqdm

from unsup_main import test_hdc_model
from unsup_ugw import save_ablation_dumbbell

import importlib.util
spec = importlib.util.spec_from_file_location("unsup_kitti_c", "unsup_kitti-c.py")
unsup_kitti_c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unsup_kitti_c)
LiDARCorruptionWrapper = unsup_kitti_c.LiDARCorruptionWrapper

MODEL_DIR = "logs/kitti_pretrain"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
NUM_CLASSES = 20
HDC_SUB_PATH = "logs/kitti_pretrain/hdc_sub_aug.pth"
LOG_DIR = "logs"

ALL_CONDITIONS = ["sunny", "snow", "fog", "motion"]
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != "sunny"]

def train_aug_hdc(ARCH, DATA, epochs=10, data_dir=None) -> AugModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=data_dir if data_dir else DATA_DIR,
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
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()
    
    # Subsample to dramatically speed up pretraining (every 4th frame)
    dataloader.dataset.scan_files = dataloader.dataset.scan_files[::4]
    dataloader.dataset.label_files = dataloader.dataset.label_files[::4]

    trainer = AugTrainer(ARCH, DATA, data_dir, LOG_DIR, MODEL_DIR, None)

    trainer.train(dataloader, trainer.model, None)

    for i in range(epochs - 1):
        trainer.retrain(dataloader, trainer.model, i+1, None)

    model: AugModel = trainer.model
    return model

def main():
    parser = argparse.ArgumentParser(description="Test Symmetric TTAug Method")
    parser.add_argument("--reinit_subclusters", action="store_true", help="Reinitialize subclusters with Symmetric method")
    parser.add_argument("--hdc_epochs", type=int, default=10, help="Number of epochs to retrain HDC model (10 by default)")
    args = parser.parse_args()

    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Truncate training sequences to [0, 1, 2, 3] (~9000 frames) so SemanticKITTI trains much faster than Waymo
    train_seqs = DATA["split"]["train"][:4]
    valid_seqs = DATA["split"]["valid"]
    DATA["split"]["train"] = train_seqs

    print("Building SemanticKITTI clean baseline parser...")
    baseline_parser = Parser(root=DATA_DIR,
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
                    shuffle_train=False)

    # Subsample training data by taking every 4th frame across the sequences
    train_dataset = baseline_parser.get_train_set().dataset
    train_dataset.scan_files = train_dataset.scan_files[::4]
    train_dataset.label_files = train_dataset.label_files[::4]

    # Subsample validation data by taking every 10th frame (1Hz) to speed up ablation testing
    valid_dataset = baseline_parser.validloader.dataset
    valid_dataset.scan_files = valid_dataset.scan_files[::10]
    valid_dataset.label_files = valid_dataset.label_files[::10]

    val_loaders = {"sunny": DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])}
    train_loaders = {}
    
    ARCH["train"]["workers"] = 0 
    print("Building SemanticKITTI-C per-condition loaders (Severity 3)...")
    for cond in ADVERSE_CONDITIONS:
        train_wrapper = LiDARCorruptionWrapper(baseline_parser.get_train_set().dataset, corruption_type=cond, severity=3)
        train_loaders[cond] = DataLoader(train_wrapper, batch_size=1, shuffle=True, num_workers=0)
        
        val_wrapper = LiDARCorruptionWrapper(baseline_parser.validloader.dataset, corruption_type=cond, severity=3)
        val_loaders[cond] = DataLoader(val_wrapper, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])

    update_methods = [
        {"name": "Baseline", "method": "inference_update", "is_active": False},
        {"name": "Symmetric TTAug", "method": "inference_update_symmetric"}
    ]

    ablation_histories = []
    
    if args.hdc_epochs > 0:
        print(f"\nRetraining Symmetric HDC prototypes for {args.hdc_epochs} epochs...")
        ARCH["train"]["batch_size"] = 6
        model_retrained = train_aug_hdc(ARCH, DATA, epochs=args.hdc_epochs, data_dir=DATA_DIR)
        torch.save(model_retrained.state_dict(), HDC_SUB_PATH)
        args.reinit_subclusters = True 

    model_base = AugModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    
    loaded_obj = torch.load(HDC_SUB_PATH, map_location=device, weights_only=False)
    if isinstance(loaded_obj, torch.nn.Module):
        model_base.load_state_dict(loaded_obj.state_dict(), strict=False)
        torch.save(loaded_obj.state_dict(), HDC_SUB_PATH)
    else:
        model_base.load_state_dict(loaded_obj, strict=False)
    model_base.to(device)
    
    if args.reinit_subclusters:
        print("\nReinitializing subclusters with Symmetric Bundled method...")
        sunny_train_loader = DataLoader(baseline_parser.get_train_set().dataset, batch_size=6, shuffle=True, num_workers=ARCH["train"]["workers"])
        model_base.eval()
        model_base.init_subclusters(sunny_train_loader)
        
        print(f"Saving updated model weights to {HDC_SUB_PATH}...")
        torch.save(model_base.state_dict(), HDC_SUB_PATH)
    
    print("\nEvaluating baseline on sunny...")
    acc_sunny, miou_sunny = test_hdc_model(model_base, val_loaders["sunny"])
    sunny_baseline = {"acc": acc_sunny, "miou": miou_sunny}
    print(f"Baseline Sunny - acc: {acc_sunny:.4f} mIoU: {miou_sunny:.4f}")
    condition_baselines = {}

    for cfg in update_methods:
        history = {
            "name": cfg["name"],
            "steps_labels": [],
            "conditions": [],
            "acc_pairs": [],
            "miou_pairs": [],
            "stats": {}
        }

        for cond in ADVERSE_CONDITIONS:
            if cond not in train_loaders:
                continue

            print(f"\n{'='*60}")
            print(f"Condition: [{cond.upper()}] | Method: {cfg['name']}")

            val_loader_for_cond = val_loaders.get(cond, next(iter(val_loaders.values())))

            if not cfg.get("is_active", True):
                continue
                
            if cond not in condition_baselines:
                print(f"    Evaluating baseline metrics for condition: {cond}...")
                b_acc, b_miou, b_stats = test_hdc_model(model_base, val_loader_for_cond, return_detailed=True)
                condition_baselines[cond] = (b_acc, b_miou, b_stats)
                
            acc_pre, miou_pre, stats_pre = condition_baselines[cond]
            print(f"    Pre  - acc: {acc_pre:.4f}  mIoU: {miou_pre:.4f}")

            model = AugModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            model.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device), strict=False)
            model.to(device)
            model.train()
            
            update_fn = getattr(model, cfg["method"]) if hasattr(model, cfg["method"]) else getattr(model, "inference_update")
            
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
                
                kwargs = {}
                
                if proj_in.shape[1] > 0:
                    update_fn(
                        proj_in.to(device),
                        learning_rate=0.001,
                        threshold=0.80,
                        **kwargs
                    )
                pbar.update(1)
            pbar.close()

            acc_post, miou_post, stats_post = test_hdc_model(model, val_loader_for_cond, return_detailed=True)
            print(f"    Post - acc: {acc_post:.4f}  mIoU: {miou_post:.4f}  Δ mIoU: {miou_post - miou_pre:+.4f}")

            history["steps_labels"].append(f"{cond.capitalize()}")
            history["conditions"].append(cond)
            history["acc_pairs"].append((acc_pre, acc_post))
            history["miou_pairs"].append((miou_pre, miou_post))
            history["stats"][cond] = {
                "pre": stats_pre,
                "post": stats_post,
                "delta_miou": miou_post - miou_pre
            }

        ablation_histories.append(history)

    log_path = f"ablation_log_ttaug_sym_{int(time.time())}.json"
    print(f"\nSaving detailed stats to {log_path}...")
    with open(log_path, 'w') as f:
        json.dump(ablation_histories, f, indent=4)

    save_ablation_dumbbell(ablation_histories, sunny_baseline=sunny_baseline, file_suffix="_ttaug_symmetric")

if __name__ == "__main__":
    main()

import argparse
import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import inspect

from dataset.kitti.parser import Parser
from modules.aug_model import AugModel
from unsup_main import test_hdc_model

unsup_kitti_c = importlib.import_module("unsup_kitti-c")
LiDARCorruptionWrapper = unsup_kitti_c.LiDARCorruptionWrapper

MODEL_DIR = "logs/kitti_pretrain"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
NUM_CLASSES = 17
CONFIG_PATH = "config/arch/senet-2048p.yml"
LABELS_PATH = "config/labels/semantic-kitti-all.yaml"
HDC_SUB_PATH = os.path.join(MODEL_DIR, "hdc_sub.pth")
SAVE_DIR = "logs/diagnostics"

def build_model(ARCH, device, state_dict):
    model = AugModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    if "subclusters" in state_dict:
        new_size = state_dict["subclusters"].shape[0]
        if model.subclusters.shape[0] != new_size:
            model.subclusters = torch.nn.Parameter(torch.zeros(new_size, model.hd_dim, device=device))
    model.load_state_dict(state_dict, strict=False)
    
    # Initialize or zero proto_momentum for original inference_update
    if not hasattr(model, 'proto_momentum'):
        model.register_buffer('proto_momentum', torch.zeros_like(model.classify.weight))
    else:
        model.proto_momentum.zero_()
        
    model.to(device)
    return model

def run_condition(model_base, ARCH, device, raw_train_dataset, valid_dataset,
                   cond, severity, method_name, method_kwargs={}, cached_b_acc=None, cached_b_miou=None):
    model = build_model(ARCH, device, model_base.state_dict())
    model.eval()

    val_target = LiDARCorruptionWrapper(valid_dataset, corruption_type=cond, severity=severity)
    val_loader = DataLoader(val_target, batch_size=1, shuffle=False, num_workers=0)

    if cached_b_acc is not None and cached_b_miou is not None:
        b_acc, b_miou = cached_b_acc, cached_b_miou
    else:
        b_acc, b_miou = test_hdc_model(model_base, val_loader)

    if method_name is not None:
        target = LiDARCorruptionWrapper(raw_train_dataset, corruption_type=cond, severity=severity)
        train_loader = DataLoader(target, batch_size=1, shuffle=False, num_workers=0)
        method_func = getattr(model, method_name)
        for batch in tqdm(train_loader, desc=f"Adapting [{cond} sev{severity} | {method_name}]"):
            proj_in = batch[0].to(device)
            proj_xyz = batch[10].to(device) if len(batch) > 10 else None
            if proj_in.shape[1] == 0:
                continue
                
            sig = inspect.signature(method_func)
            has_xyz = 'proj_xyz' in sig.parameters
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            
            if has_xyz or has_kwargs:
                method_func(proj_in, proj_xyz=proj_xyz, **method_kwargs)
            else:
                method_func(proj_in, **method_kwargs)

    acc, miou = test_hdc_model(model, val_loader)
    
    firing_log = getattr(model, '_firing_log', None)
    return b_acc, b_miou, acc, miou, firing_log

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ARCH = yaml.safe_load(open(CONFIG_PATH))
    DATA = yaml.safe_load(open(LABELS_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs = DATA["split"]["train"][:1]
    valid_seqs = DATA["split"]["valid"]

    baseline_parser = Parser(
        root=DATA_DIR, train_sequences=train_seqs, valid_sequences=valid_seqs,
        test_sequences=None, labels=DATA["labels"], color_map=DATA["color_map"],
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=0, gt=True, shuffle_train=False,
    )
    raw_train_dataset = baseline_parser.get_train_set().dataset
    raw_train_dataset.scan_files = raw_train_dataset.scan_files[::4]
    raw_train_dataset.label_files = raw_train_dataset.label_files[::4]

    valid_dataset = baseline_parser.validloader.dataset
    valid_dataset.scan_files = valid_dataset.scan_files[::10]
    valid_dataset.label_files = valid_dataset.label_files[::10]

    loaded_obj = torch.load(HDC_SUB_PATH, map_location=device, weights_only=False)
    state_dict = loaded_obj.state_dict() if isinstance(loaded_obj, torch.nn.Module) else loaded_obj
    model_base = build_model(ARCH, device, state_dict)
    model_base.eval()

    results = []
    
    conditions = ["snow", "fog", "motion"]
    for cond in conditions:
        severity = 2
        cached_b_acc, cached_b_miou = None, None
        
        methods_to_test = [
            (None, "Frozen Baseline", {}),
            ("inference_update", "Density Baseline", {"learning_rate": 0.001, "distance_sensitivity": 3.0, "thresholds": [0.45, 0.80]}),
            ("inference_update_soft_consensus", "Exp A (LR=0.001)", {"learning_rate": 0.001}),
            ("inference_update_soft_consensus", "Exp A (LR=0.01)", {"learning_rate": 0.01}),
        ]
        
        for method_name, label, method_kwargs in methods_to_test:
            b_acc, b_miou, acc, miou, firing_log = run_condition(
                model_base, ARCH, device, raw_train_dataset, valid_dataset,
                cond=cond, severity=severity, method_name=method_name, method_kwargs=method_kwargs,
                cached_b_acc=cached_b_acc, cached_b_miou=cached_b_miou
            )
            cached_b_acc, cached_b_miou = b_acc, b_miou
            
            avg_fire = sum(firing_log)/len(firing_log) if firing_log else 0.0
            print(f"[{cond} sev{severity} | {label}] baseline mIoU={b_miou:.4f} -> {miou:.4f} "
                  f"(delta {miou - b_miou:+.4f}) | fire_rate={avg_fire:.2%}")
            
            result_entry = {
                "test": "compare_expA_vs_baseline", "condition": cond, "severity": severity, "method": label,
                "acc_pair": [b_acc, acc], "miou_pair": [b_miou, miou],
            }
            if firing_log:
                result_entry["firing_rate"] = avg_fire
            results.append(result_entry)

    out_path = os.path.join(SAVE_DIR, "next_compare.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()

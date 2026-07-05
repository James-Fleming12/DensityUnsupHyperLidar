import argparse
import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib

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
    model.to(device)
    return model


def run_condition(model_base, ARCH, device, raw_train_dataset, valid_dataset,
                   cond, severity, method_name, cached_b_acc=None, cached_b_miou=None):
    model = build_model(ARCH, device, model_base.state_dict())
    model.eval()  # IMPORTANT: always eval, never .train() -- see BN-contamination note

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
            method_func(proj_in, proj_xyz=proj_xyz)

    acc, miou = test_hdc_model(model, val_loader)
    
    firing_log = getattr(model, '_firing_log', None)
    return b_acc, b_miou, acc, miou, firing_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["cross_sensor_sweep", "conditional_opp", "broad_exp_a", "all"], required=True)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    ARCH = yaml.safe_load(open(CONFIG_PATH))
    DATA = yaml.safe_load(open(LABELS_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs = DATA["split"]["train"][:4]
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
    if args.test == "all":
        tests_to_run = ["cross_sensor_sweep", "conditional_opp", "broad_exp_a"]
    else:
        tests_to_run = [args.test]

    if "cross_sensor_sweep" in tests_to_run:
        for severity in [3, 4, 5]:
            cached_b_acc, cached_b_miou = None, None
            for method_name, label in [(None, "Frozen Baseline"),
                                        ("inference_update_soft_consensus", "Exp A")]:
                b_acc, b_miou, acc, miou, firing_log = run_condition(
                    model_base, ARCH, device, raw_train_dataset, valid_dataset,
                    cond="cross_sensor", severity=severity, method_name=method_name,
                    cached_b_acc=cached_b_acc, cached_b_miou=cached_b_miou
                )
                cached_b_acc, cached_b_miou = b_acc, b_miou
                
                avg_fire = sum(firing_log)/len(firing_log) if firing_log else 0.0
                print(f"[sev {severity} | {label}] baseline mIoU={b_miou:.4f} -> {miou:.4f} "
                      f"(delta {miou - b_miou:+.4f}) | fire_rate={avg_fire:.2%}")
                
                result_entry = {
                    "test": "cross_sensor_sweep", "severity": severity, "method": label,
                    "acc_pair": [b_acc, acc], "miou_pair": [b_miou, miou],
                }
                if firing_log:
                    result_entry["firing_rate"] = avg_fire
                results.append(result_entry)

    if "conditional_opp" in tests_to_run:
        for cond in ["snow", "fog"]:
            cached_b_acc, cached_b_miou = None, None
            for method_name, label in [(None, "Frozen Baseline"),
                                        ("inference_update_opp", "Plain OPP"),
                                        ("inference_update_conditional_opp", "Conditional OPP")]:
                b_acc, b_miou, acc, miou, firing_log = run_condition(
                    model_base, ARCH, device, raw_train_dataset, valid_dataset,
                    cond=cond, severity=3, method_name=method_name,
                    cached_b_acc=cached_b_acc, cached_b_miou=cached_b_miou
                )
                cached_b_acc, cached_b_miou = b_acc, b_miou
                
                avg_fire = sum(firing_log)/len(firing_log) if firing_log else 0.0
                print(f"[{cond} | {label}] baseline mIoU={b_miou:.4f} -> {miou:.4f} "
                      f"(delta {miou - b_miou:+.4f}) | fire_rate={avg_fire:.2%}")
                
                result_entry = {
                    "test": "conditional_opp", "condition": cond, "method": label,
                    "acc_pair": [b_acc, acc], "miou_pair": [b_miou, miou],
                }
                if firing_log:
                    result_entry["firing_rate"] = avg_fire
                results.append(result_entry)

    if "broad_exp_a" in tests_to_run:
        # Note: Added realistic KITTI-C corruption names
        all_conditions = ["snow", "fog", "cross_sensor", "motion", "beam", "crosstalk", "echo"]
        for cond in all_conditions:
            for severity in [2, 3, 4]:
                cached_b_acc, cached_b_miou = None, None
                for method_name, label in [(None, "Frozen Baseline"),
                                            ("inference_update_soft_consensus", "Exp A"),
                                            ("inference_update_safe_consensus", "Safe Exp A")]:
                    b_acc, b_miou, acc, miou, firing_log = run_condition(
                        model_base, ARCH, device, raw_train_dataset, valid_dataset,
                        cond=cond, severity=severity, method_name=method_name,
                        cached_b_acc=cached_b_acc, cached_b_miou=cached_b_miou
                    )
                    cached_b_acc, cached_b_miou = b_acc, b_miou
                    
                    avg_fire = sum(firing_log)/len(firing_log) if firing_log else 0.0
                    print(f"[{cond} sev{severity} | {label}] baseline mIoU={b_miou:.4f} -> {miou:.4f} "
                          f"(delta {miou - b_miou:+.4f}) | fire_rate={avg_fire:.2%}")
                    
                    result_entry = {
                        "test": "broad_exp_a", "condition": cond, "severity": severity, "method": label,
                        "acc_pair": [b_acc, acc], "miou_pair": [b_miou, miou],
                    }
                    if firing_log:
                        result_entry["firing_rate"] = avg_fire
                    results.append(result_entry)

    out_path = os.path.join(SAVE_DIR, f"next_round_{args.test}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

# python unsup_classify.py --dataset kradar --data_dir /path/to/kradar
# python unsup_classify.py --dataset v2xr --data_dir /path/to/v2xr

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Optional, Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torchdata
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from modules.HDC_cl import ClassificationDensityModel, DensityClassifier, PointPillarEncoder, RadarTensorEncoder

from dataset.classify import KRadarDataset, kradar_collate, V2XRDataset, v2xr_collate

SAVE_PATH = "logs/hdc_cls.pth"
SAVE_SUB_PATH = "logs/hdc_cls_sub.pth"

CONDITION_COLORS = {
    "sunny":"#F5C518",
    "clear":"#F5C518",
    "rain":"#4C9BE8",
    "sleet":"#6EC6E8",
    "fog":"#A0A0A0",
    "night":"#7B4EA0",
    "snow":"#A8D8EA",
}
DEFAULT_COLOR = "#AAAAAA"

def get_condition_loaders(
    dataset_cls,
    dataset_kwargs: dict,
    conditions: List[str],
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
    collate_fn=None,
) -> Dict[str, torchdata.DataLoader]:
    """
    Build one DataLoader per weather condition.
    Returns only conditions that have at least one sample.
    """
    loaders: Dict[str, torchdata.DataLoader] = {}
    for cond in conditions:
        kw = {**dataset_kwargs, "conditions": [cond]}
        ds = dataset_cls(**kw)
        if len(ds) == 0:
            print(f"  [get_condition_loaders] '{cond}' — 0 samples, skipping.")
            continue
        loaders[cond] = torchdata.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"  [get_condition_loaders] '{cond}' — {len(ds)} samples")
    return loaders

def build_model(
    dataset: str,
    num_classes: int,
    device: torch.device,
    subcluster_type: str = "bipolar",
) -> ClassificationDensityModel:
    """
    Instantiate the correct backbone + ClassificationDensityModel.

    Parameters
    ----------
    dataset: 'v2xr' | 'kradar'
    num_classes: total number of object classes
    device: target device
    subcluster_type: 'bipolar' | 'continuous'
    """
    if dataset == "v2xr":
        backbone = PointPillarEncoder(in_channels=4, bev_shape=(512, 512))
    elif dataset == "kradar":
        backbone = RadarTensorEncoder(range_bins=256, azimuth_bins=107)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'v2xr' or 'kradar'.")

    model = ClassificationDensityModel(
        backbone=backbone,
        num_classes=num_classes,
        device=device,
        hd_encoder="rp",
        max_subclusters=10,
        subcluster_type=subcluster_type,
        gauss_rp=True,
    )
    return model.to(device)

def pretrain_pipeline(
    args,
    dataset_cls,
    dataset_kwargs_base: dict,
    normal_condition: str,
    collate_fn,
    num_classes: int,
    device: torch.device,
) -> ClassificationDensityModel:
    print(f"--- Pretraining on '{normal_condition}' frames ---")

    train_kw = {**dataset_kwargs_base, "split": "train", "conditions": [normal_condition]}
    val_kw = {**dataset_kwargs_base, "split": "val",   "conditions": [normal_condition]}

    train_ds = dataset_cls(**train_kw)
    val_ds = dataset_cls(**val_kw)

    if len(train_ds) == 0:
        raise RuntimeError(f"No '{normal_condition}' training samples found.")

    train_loader = torchdata.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, drop_last=True,)
    val_loader = torchdata.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn,)

    model = build_model(args.dataset, num_classes, device, args.subcluster_type)
    trainer = DensityClassifier(
        model=model,
        num_classes=num_classes,
        device=device,
        epochs=args.epochs,
        bipolar_prototypes=args.bipolar_prototypes,
    )

    trainer.start(train_loader, val_loader)

    print("Initialising subclusters...")
    model.init_subclusters(
        dataloader=train_loader,
        max_samples_per_class=args.max_samples_per_class,
        sampling_strategy=args.sampling_strategy,
    )

    os.makedirs(os.path.dirname(SAVE_SUB_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_SUB_PATH)
    print(f"Pretraining complete — model saved to {SAVE_SUB_PATH}")

    return model

def test_model(model: ClassificationDensityModel, loader, device: torch.device):
    """
    Evaluate model accuracy over a DataLoader.
    Returns (accuracy, per_class_acc_dict).
    Mirrors test_hdc_model() from the Waymo pipeline.
    """
    model.eval()
    correct = 0
    total = 0
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            if isinstance(batch[0], dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
            else:
                inputs = batch[0].to(device)
            labels = batch[1].to(device).flatten()

            logits, _ = model(inputs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

            for cls in labels.unique():
                c = cls.item()
                mask = labels == cls
                per_class_correct[c] = per_class_correct.get(c, 0) + (preds[mask] == cls).sum().item()
                per_class_total[c] = per_class_total.get(c,   0) + mask.sum().item()

    acc = correct / total if total > 0 else 0.0
    per_class = {
        c: per_class_correct[c] / per_class_total[c]
        for c in per_class_total
        if per_class_total[c] > 0
    }
    return acc, per_class

def incremental_update_test(
    args,
    dataset_cls,
    dataset_kwargs_base: dict,
    adverse_conditions: List[str],
    normal_condition: str,
    collate_fn,
    num_classes: int,
    device: torch.device,
    seen_classes: Optional[Set[int]] = None,
):
    model = build_model(args.dataset, num_classes, device, args.subcluster_type)
    state = torch.load(SAVE_SUB_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    all_conditions = [normal_condition] + adverse_conditions
    val_loaders = get_condition_loaders(
        dataset_cls=dataset_cls,
        dataset_kwargs={**dataset_kwargs_base, "split": "val"},
        conditions=all_conditions,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    if not val_loaders:
        raise RuntimeError("No validation frames found for any condition.")

    history = {
        "steps_labels": [],
        "conditions": [],
        "acc_pairs": [],
        "novel_classes": [],
    }

    for cond in adverse_conditions:
        print(f"\n{'='*60}")
        print(f"Unsupervised Update Phase: [{cond.upper()}]")

        train_loaders = get_condition_loaders(
            dataset_cls=dataset_cls,
            dataset_kwargs={**dataset_kwargs_base, "split": "train"},
            conditions=[cond],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_fn,
        )

        if cond not in train_loaders:
            print(f"  No '{cond}' training frames — skipping.")
            continue

        novel_in_step: Set[int] = set()
        if seen_classes is not None:
            cond_classes = _collect_seen_classes(train_loaders[cond])
            novel_in_step = cond_classes - seen_classes
            if novel_in_step:
                print(f"    Novel classes detected: {sorted(novel_in_step)}")
            else:
                print(f"    No novel classes.")

        val_loader = val_loaders.get(cond) or next(iter(val_loaders.values()))

        acc_pre, _ = test_model(model, val_loader, device)
        print(f"    Pre  — acc: {acc_pre:.4f}")

        model.train()
        for batch in tqdm(train_loaders[cond], desc=f"    update [{cond}]", leave=False):
            if isinstance(batch[0], dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
            else:
                inputs = batch[0].to(device)

            model.inference_update(inputs, learning_rate=0.001, distance_sensitivity=3.0)

        acc_post, _ = test_model(model, val_loader, device)
        print(f"    Post — acc: {acc_post:.4f}  Δ acc: {acc_post - acc_pre:+.4f}")

        history["steps_labels"].append(f"Updates on {cond.capitalize()}")
        history["conditions"].append(cond)
        history["acc_pairs"].append((acc_pre, acc_post))
        history["novel_classes"].append(novel_in_step)

        if seen_classes is not None:
            seen_classes = seen_classes | _collect_seen_classes(train_loaders[cond])

    _save_dumbbell_plot(history, file_suffix=f"_{args.dataset}_condition_split")

def _collect_seen_classes(loader) -> Set[int]:
    """Return the set of class IDs present in a DataLoader's labels."""
    seen: Set[int] = set()
    for batch in loader:
        labels = batch[1].flatten().tolist()
        seen.update(int(l) for l in labels if int(l) != 255)
    return seen

def _save_dumbbell_plot(history: dict, file_suffix: str = ""):
    labels = history["steps_labels"]
    conditions = history["conditions"]
    acc_pairs = np.array(history["acc_pairs"])
    novel_classes = history.get("novel_classes", [set() for _ in labels])

    while len(novel_classes) < len(labels):
        novel_classes.append(set())

    if len(acc_pairs) == 0:
        print("No data to plot.")
        return

    fig = plt.figure(figsize=(18, max(6, len(labels) * 0.85 + 3)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    y_pos = np.arange(len(labels))
    COLOR_PRE = "#4C9BE8"
    COLOR_POST = "#E8574C"
    COLOR_NOVEL = "#FFF3CD"
    COLOR_NONE = "#F4F4F4"

    def row_bg(yi):
        if len(novel_classes[yi]) > 0:
            return COLOR_NOVEL
        cond = conditions[yi] if yi < len(conditions) else "clear"
        return CONDITION_COLORS.get(cond, DEFAULT_COLOR) + "33"

    for yi in range(len(acc_pairs)):
        ax1.axhspan(yi - 0.45, yi + 0.45, color=row_bg(yi), zorder=0, alpha=0.8)
    ax1.hlines(y_pos, acc_pairs[:, 0], acc_pairs[:, 1], color="#AAAAAA", alpha=0.6, linewidth=2, zorder=1)
    ax1.scatter(acc_pairs[:, 0], y_pos, color=COLOR_PRE,  s=130, label="Pre-Update",  zorder=3, edgecolors="white", linewidths=0.8)
    ax1.scatter(acc_pairs[:, 1], y_pos, color=COLOR_POST, s=130, label="Post-Update", zorder=3, edgecolors="white", linewidths=0.8)
    ax1.set_title("Accuracy Gain per Condition", fontsize=13, fontweight="bold", pad=10)
    ax1.set_yticks(y_pos)
    tick_labels = ax1.set_yticklabels(labels, fontsize=8)
    for tick, cond in zip(tick_labels, conditions):
        tick.set_color(CONDITION_COLORS.get(cond, "black"))
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.set_xlabel("Accuracy", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=9)

    deltas = acc_pairs[:, 1] - acc_pairs[:, 0]
    colors = [COLOR_POST if d >= 0 else "#E8574C" for d in deltas]
    ax2.barh(y_pos, deltas, color=colors, alpha=0.8)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Δ Accuracy per Condition", fontsize=13, fontweight="bold", pad=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.set_xlabel("Δ Accuracy", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    cond_patches = [mpatches.Patch(color=CONDITION_COLORS.get(c, DEFAULT_COLOR), label=c.capitalize()) for c in dict.fromkeys(conditions)]
    ax1.legend(
        handles=cond_patches,
        title="Condition", loc="upper left",
        fontsize=8, title_fontsize=8,
        bbox_to_anchor=(0, -0.06), ncol=len(cond_patches),
        frameon=True, framealpha=0.9,
    )

    plt.suptitle("Impact of Incremental Unsupervised Inference Updates", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = f"incremental_dumbbell_results{file_suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dumbbell plot saved to {out}")

def parse_args():
    p = argparse.ArgumentParser(description="Unsupervised HDC classification pipeline")

    p.add_argument("--dataset", choices=["v2xr", "kradar"], required=True, help="Which dataset to use")
    p.add_argument("--data_dir", required=True, help="Root directory of the dataset")

    p.add_argument("--epochs", type=int, default=20, help="Number of HDC refinement epochs")
    p.add_argument("--batch_size",type=int, default=8)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--subcluster_type", default="bipolar", choices=["bipolar", "continuous"])
    p.add_argument("--bipolar_prototypes", action="store_true", help="Binarise prototype weights after each training pass")
    p.add_argument("--max_samples_per_class", type=int, default=4000)
    p.add_argument("--sampling_strategy", default="diverse", choices=["random", "diverse", "fps"])
    p.add_argument("--skip_pretrain", action="store_true", help="Skip pretraining and load from checkpoint")

    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "v2xr":
        dataset_cls = V2XRDataset
        collate_fn = v2xr_collate
        normal_condition = "sunny"
        adverse_conditions = ["rain", "night"]
        num_classes = max(V2XRDataset.DEFAULT_LABEL_MAP.values()) + 1

        dataset_kwargs_base = {"data_dir":  args.data_dir}
    elif args.dataset == "kradar":
        dataset_cls = KRadarDataset
        collate_fn = kradar_collate
        normal_condition = "clear"
        adverse_conditions = ["rain", "sleet", "snow"]
        num_classes = max(KRadarDataset.DEFAULT_LABEL_MAP.values()) + 1

        dataset_kwargs_base = {"data_dir":  args.data_dir}
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset: {args.dataset}  |  classes: {num_classes}  |  normal: {normal_condition}  |  adverse: {adverse_conditions}")

    if args.skip_pretrain:
        print(f"Skipping pretraining — loading from {SAVE_SUB_PATH}")
        model = build_model(args.dataset, num_classes, device, args.subcluster_type)
        model.load_state_dict(torch.load(SAVE_SUB_PATH, map_location=device))
        model.to(device)
        normal_loader = torchdata.DataLoader(
            dataset_cls(**{**dataset_kwargs_base, "split": "train", "conditions": [normal_condition]}),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn,
        )
        seen_classes = _collect_seen_classes(normal_loader)
    else:
        model = pretrain_pipeline(
            args=args,
            dataset_cls=dataset_cls,
            dataset_kwargs_base=dataset_kwargs_base,
            normal_condition=normal_condition,
            collate_fn=collate_fn,
            num_classes=num_classes,
            device=device,
        )
        normal_loader = torchdata.DataLoader(
            dataset_cls(**{**dataset_kwargs_base, "split": "train", "conditions": [normal_condition]}),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn,
        )
        seen_classes = _collect_seen_classes(normal_loader)

    incremental_update_test(
        args=args,
        dataset_cls=dataset_cls,
        dataset_kwargs_base=dataset_kwargs_base,
        adverse_conditions=adverse_conditions,
        normal_condition=normal_condition,
        collate_fn=collate_fn,
        num_classes=num_classes,
        device=device,
        seen_classes=seen_classes,
    )

if __name__ == "__main__":
    main()
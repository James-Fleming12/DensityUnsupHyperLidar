from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torchdata
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from dataset.ai_motive import (
    AiMotiveDataset,
    AiMotiveParser,
    CLASS_MAP,
    ALL_CONDITIONS,
    NORMAL_CONDITION,
    ADVERSE_CONDITIONS,
    NUM_CLASSES,
    _parser_collate
)

from modules.HDC_cl import PointPillarEncoder
from modules.HDC_utils import DensityModel
from modules.Basic_HD import DensityTrainer
from unsup_main import train_hdc, test_hdc_model

DATA_DIR = "/path/to/aimotive"
MODEL_DIR = "logs"
LOG_DIR = "logs"
HDC_SAVE_PATH = "logs/aimotive_hdc.pth"
HDC_SUB_PATH = "logs/aimotive_hdc_sub.pth"

BATCH_SIZE = 4
WORKERS = 4
SUBCLUSTER_TYPE = "continuous"
MAX_HDC_EPOCHS = 5
USE_MENT = False

CONDITION_COLORS = {
    "daytime": "#F5C518",
    "night":   "#7B4EA0",
    "rainy":   "#4C9BE8",
}
DEFAULT_COLOR = "#AAAAAA"

def build_model(num_classes: int, device: torch.device, subcluster_type: str = "continuous") -> DensityModel:
    """
    Build a DensityModel with a PointPillarEncoder backbone.

    PointPillarEncoder produces (B, 128) embeddings. We wrap it to output
    (B, 128, 1, 1) so DensityModel.encode()'s permute+reshape gives
    (B*1*1, 128) = (B, 128) — one HV per sample.
    """
    backbone = PointPillarEncoder(in_channels=4, bev_shape=(512, 512))

    class _WrappedBackbone(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, x, only_feat=False):
            feat = self.enc(x)
            return feat.unsqueeze(-1).unsqueeze(-1)

    ARCH = {
        "train": {
            "pipeline": "_aimotive",
            "aux_loss": False,
            "act": "SiLU",
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "epsilon_w": 0.001,
        },
        "post": {"KNN": {"use": False, "params": {}}},
        "dataset": {"sensor": {}, "max_points": 35000},
    }

    model = DensityModel(
        ARCH=ARCH,
        modeldir=MODEL_DIR,
        hd_encoder="rp",
        num_levels=0,
        randomness=0.0,
        num_classes=num_classes,
        device=device,
        subcluster_type=subcluster_type,
    )
    model.net = _WrappedBackbone(backbone).to(device)
    model.net.eval()
    return model.to(device)

def get_condition_loaders(split: str, conditions: List[str], batch_size: int = BATCH_SIZE, shuffle: bool = False, workers: int = WORKERS, val_fraction: float = 0.2,) -> Dict[str, torchdata.DataLoader]:
    """
    Build one DataLoader per weather condition.
    Returns only conditions that have at least one sample.
    Yields the 15-tuple format via _parser_collate so it is compatible
    with both DensityTrainer and the update loops below.
    """
    loaders: Dict[str, torchdata.DataLoader] = {}
    for cond in conditions:
        ds = AiMotiveDataset(
            root=DATA_DIR,
            split=split,
            conditions=[cond],
            val_fraction=val_fraction,
        )
        if len(ds) == 0:
            print(f"  [get_condition_loaders] '{cond}' — 0 frames, skipping.")
            continue
        loaders[cond] = torchdata.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=_parser_collate,
            drop_last=False,
        )
        print(f"  [get_condition_loaders] '{cond}' — {len(ds)} frames")
    return loaders

def run_ent_minimization(model: DensityModel, target_loader, epochs: int = 3, lr: float = 1e-5):
    """
    Entropy minimization on unlabelled target-domain frames.
    Only updates model.net — HDC classify weights are untouched.
    Call trainer.reaccumulate_prototypes() on the source loader afterward.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.net.parameters(), lr=lr)

    print(f"--- MinEnt: {epochs} epoch(s) on target data ---")
    for epoch in range(epochs):
        total_entropy = 0.0
        for proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _ in \
                tqdm(target_loader, desc=f"MinEnt epoch {epoch + 1}"):
            if isinstance(proj_in, dict):
                proj_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proj_in.items()}
            else:
                proj_in = proj_in.to(device)
            ent = model.entropy_minimization_step(proj_in, optimizer)
            total_entropy += ent
        print(f"  Epoch {epoch + 1}  mean entropy: {total_entropy / len(target_loader):.4f}")

    model.net.eval()

def test_model(model: DensityModel, loader, device: torch.device):
    """
    Evaluate accuracy over a condition DataLoader.
    Returns (accuracy, per_class_acc_dict).
    """
    model.eval()
    correct = 0
    total = 0
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}

    with torch.no_grad():
        for proj_in, proj_mask, proj_labels, *_ in loader:
            if isinstance(proj_in, dict):
                proj_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proj_in.items()}
            else:
                proj_in = proj_in.to(device)

            if proj_labels is None:
                continue
            labels = proj_labels.to(device).flatten()

            enc, _, _ = model.encode(proj_in)
            logits = model.get_predictions(enc)
            preds = logits.argmax(dim=1)

            if preds.shape[0] != labels.shape[0]:
                n_samples = labels.shape[0]
                preds = preds.reshape(n_samples, -1).mode(dim=1).values

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
        for c in per_class_total if per_class_total[c] > 0
    }
    print(f"  accuracy: {acc:.4f}  ({correct}/{total})")
    return acc, per_class

def _make_arch() -> dict:
    """Minimal ARCH dict consumed by DensityTrainer.__init__."""
    return {
        "train": {
            "pipeline": "_aimotive",
            "aux_loss": False,
            "act": "SiLU",
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "epsilon_w": 0.001,
        },
        "post": {"KNN": {"use": False, "params": {}}},
        "dataset": {"sensor": {}, "max_points": 35000},
    }


def _make_data_cfg(parser: AiMotiveParser) -> dict:
    n = parser.num_classes
    return {
        "split": {"train": [], "valid": []},
        "labels": {i: name for name, i in CLASS_MAP.items()},
        "color_map": {i: [128, 128, 128] for i in range(n)},
        "learning_map": {i: i for i in range(n)},
        "learning_map_inv": {i: i for i in range(n)},
        "learning_ignore": {i: False for i in range(n)},
        "content": {i: 1.0 / n for i in range(n)},
    }

def _parser_collate_concat(batch):
    return _parser_collate(batch)

def pretrain_pipeline(device: torch.device, use_ment: bool = True):
    print(f"--- Pretraining on '{NORMAL_CONDITION}' frames (use_ment={use_ment}) ---")

    parser = AiMotiveParser(
        root=DATA_DIR,
        conditions=[NORMAL_CONDITION],
        batch_size=BATCH_SIZE,
        workers=WORKERS,
    )

    trainer = DensityTrainer(
        ARCH=_make_arch(),
        DATA=_make_data_cfg(parser),
        datadir=DATA_DIR,
        logdir=LOG_DIR,
        modeldir=MODEL_DIR,
        logger=None,
        bipolar_prototypes=False,
        bipolar_subclusters=(SUBCLUSTER_TYPE == "bipolar"),
    )
    model = build_model(NUM_CLASSES, device, SUBCLUSTER_TYPE)
    trainer.model = model

    print("Training HDC Density Model (daytime only)...")
    trainer.start()
    torch.save(model.state_dict(), HDC_SAVE_PATH)
    print(f"HDC checkpoint saved to {HDC_SAVE_PATH}")

    daytime_loaders = get_condition_loaders(
        split="train",
        conditions=[NORMAL_CONDITION],
        batch_size=BATCH_SIZE,
        shuffle=True,
        workers=WORKERS,
    )

    if NORMAL_CONDITION not in daytime_loaders:
        raise RuntimeError("No daytime training frames found.")

    if use_ment:
        adverse_loaders = get_condition_loaders(
            split="train",
            conditions=ADVERSE_CONDITIONS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            workers=WORKERS,
        )

        if not adverse_loaders:
            print("  Warning: no adverse frames found, skipping MinEnt.")
        else:
            target_dataset = torchdata.ConcatDataset(
                [loader.dataset for loader in adverse_loaders.values()]
            )
            target_loader = torchdata.DataLoader(
                target_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=WORKERS,
                collate_fn=_parser_collate_concat,
                drop_last=True,
            )

            run_ent_minimization(model, target_loader, epochs=3, lr=1e-5)

            trainer.reaccumulate_prototypes(daytime_loaders[NORMAL_CONDITION])

            print("Re-running retrain epochs after MinEnt...")
            for epoch in range(1, MAX_HDC_EPOCHS + 1):
                trainer.retrain(daytime_loaders[NORMAL_CONDITION], model, epoch, trainer.logger)
    else:
        print("MinEnt disabled — skipping to subcluster initialisation.")

    print("Initializing Subclusters...")
    model.init_subclusters(daytime_loaders[NORMAL_CONDITION])

    os.makedirs(os.path.dirname(HDC_SUB_PATH), exist_ok=True)
    torch.save(model.state_dict(), HDC_SUB_PATH)
    print(f"Pretraining complete. Model saved to {HDC_SUB_PATH}")

    return model

def incremental_update_test(device: torch.device):
    model = build_model(NUM_CLASSES, device, SUBCLUSTER_TYPE)
    model.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device))
    model.to(device)

    print("Building per-condition validation loaders...")
    val_loaders = get_condition_loaders(
        split="val",
        conditions=ALL_CONDITIONS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        workers=WORKERS,
    )

    if not val_loaders:
        raise RuntimeError("No validation frames found for any condition.")

    train_loaders = get_condition_loaders(
        split="train",
        conditions=ADVERSE_CONDITIONS,
        batch_size=1,
        shuffle=True,
        workers=WORKERS,
    )

    if not train_loaders:
        print("  No adverse condition frames found — skipping incremental update.")
        return

    history = {
        "steps_labels": [],
        "conditions": [],
        "acc_pairs": [],
    }

    print(f"--- Incremental Evaluation: Unsupervised Updates on Adverse Conditions ---")

    for cond in ADVERSE_CONDITIONS:
        if cond not in train_loaders:
            continue

        print(f"\n{'='*60}")
        print(f"Unsupervised Update Phase: [{cond.upper()}]")

        step_label = f"Updates on {cond.capitalize()}"
        val_loader = val_loaders.get(cond) or next(iter(val_loaders.values()))

        acc_pre, _ = test_model(model, val_loader, device)
        print(f"    Pre  — acc: {acc_pre:.4f}")

        model.train()
        for proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _ in \
                tqdm(train_loaders[cond], desc=f"    update [{cond}]", leave=False):
            if isinstance(proj_in, dict):
                proj_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in proj_in.items()}
            else:
                proj_in = proj_in.to(device)
            model.inference_update(proj_in, learning_rate=0.001, distance_sensitivity=3.0)

        acc_post, _ = test_model(model, val_loader, device)
        print(f"    Post — acc: {acc_post:.4f}  Δ acc: {acc_post - acc_pre:+.4f}")

        history["steps_labels"].append(step_label)
        history["conditions"].append(cond)
        history["acc_pairs"].append((acc_pre, acc_post))

    save_dumbbell_plot(history, file_suffix="_aimotive_condition_split")

def save_dumbbell_plot(history: dict, file_suffix: str = ""):
    labels = history["steps_labels"]
    conditions = history["conditions"]
    acc_pairs = np.array(history["acc_pairs"])

    if len(acc_pairs) == 0:
        print("No data to plot.")
        return

    COLOR_PRE = "#4C9BE8"
    COLOR_POST = "#E8574C"

    fig = plt.figure(figsize=(18, max(6, len(labels) * 0.85 + 3)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    y_pos = np.arange(len(labels))
    
    def row_bg(yi):
        cond = conditions[yi] if yi < len(conditions) else NORMAL_CONDITION
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

    cond_patches = [mpatches.Patch(color=CONDITION_COLORS.get(c, DEFAULT_COLOR), label=c.capitalize()) for c in dict.fromkeys(conditions)]
    ax1.legend(
        handles=cond_patches,
        title="Condition", loc="upper left",
        fontsize=8, title_fontsize=8,
        bbox_to_anchor=(0, -0.06), ncol=len(cond_patches),
        frameon=True, framealpha=0.9,
    )

    deltas = acc_pairs[:, 1] - acc_pairs[:, 0]
    bar_colors = [COLOR_POST if d >= 0 else COLOR_PRE for d in deltas]
    ax2.barh(y_pos, deltas, color=bar_colors, alpha=0.8)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Δ Accuracy per Condition", fontsize=13, fontweight="bold", pad=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.set_xlabel("Δ Accuracy", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Impact of Incremental Unsupervised Inference Updates — aiMotive", fontsize=15, fontweight="bold", y=1.01,)
    plt.tight_layout()

    out = f"incremental_dumbbell_results{file_suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dumbbell plot saved to {out}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset root: {DATA_DIR}")
    print(f"MinEnt: {'enabled' if USE_MENT else 'disabled'}")

    model = pretrain_pipeline(device, use_ment=USE_MENT)

    incremental_update_test(device)

if __name__ == "__main__":
    main()
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.utils.data as torchdata
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import yaml

from dataset.ai_motive import (
    AiMotiveDensityTrainer,
    AiMotiveDataset,
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

DATA_DIR = "/mnt/bravo/jmfleming/ai_motive"
MODEL_DIR = "logs"
LOG_DIR = "logs"
HDC_SAVE_PATH = "logs/aimotive_hdc.pth"
HDC_SUB_PATH = "logs/aimotive_hdc_sub.pth"

BATCH_SIZE = 4
WORKERS = 4
SUBCLUSTER_TYPE = "continuous"
USE_MENT = False

FEATURE_EXTRACTOR_EPOCHS = 50
MAX_HDC_EPOCHS = 8
UNSUP_EPOCHS = 5

CONDITION_COLORS = {
    "highway":"#F5C518",
    "urban": "#E8572A",
    "night": "#7B4EA0",
    "rain": "#4C9BE8",
}
DEFAULT_COLOR = "#AAAAAA"
SOURCE_DOMAIN = "highway"

def build_model(num_classes, device, ARCH, subcluster_type="continuous"):
    ARCH["train"]["pipeline"] = "pointpillar"

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

    ckpt_path = os.path.join(MODEL_DIR, "aimotive_feature_extractor.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.net.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"  Loaded feature extractor from {ckpt_path}")

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

def _parser_collate_concat(batch):
    return _parser_collate(batch)

def train_feature_extractor(model_arch, train_loader, device, epochs=10):
    backbone = PointPillarEncoder(in_channels=4, bev_shape=(512, 512)).to(device)
    backbone.train()

    head = torch.nn.Linear(128, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for proj_in, _, proj_labels, *_ in tqdm(train_loader, desc=f"FE Epoch {epoch+1}"):
            if isinstance(proj_in, dict):
                proj_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proj_in.items()}
            else:
                proj_in = proj_in.to(device)

            proj_labels = proj_labels.to(device).flatten()
            optimizer.zero_grad()

            feat = backbone(proj_in)
            logits = head(feat)
            loss = criterion(logits, proj_labels)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            valid_mask = proj_labels != -1
            if valid_mask.any():
                correct += (logits.argmax(1)[valid_mask] == proj_labels[valid_mask]).sum().item()
                total += valid_mask.sum().item()

        epoch_loss = total_loss / len(train_loader)
        print(f"  FE Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {correct/total if total else 0:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({"state_dict": backbone.state_dict()}, os.path.join(MODEL_DIR, "aimotive_feature_extractor.pth"))
            print(f"  Saved best checkpoint (loss {best_loss:.4f})")

    backbone.eval()
    print("Feature extractor training complete.\n")

def pretrain_pipeline(device: torch.device, ARCH: dict, use_ment: bool = True):
    print(f"--- Pretraining on '{NORMAL_CONDITION}' frames (use_ment={use_ment}) ---")

    daytime_loaders = get_condition_loaders(
        split="train",
        conditions=[NORMAL_CONDITION],
        batch_size=BATCH_SIZE,
        shuffle=True,
        workers=WORKERS,
    )

    if NORMAL_CONDITION not in daytime_loaders:
        raise RuntimeError("No daytime training frames found.")

    train_feature_extractor(ARCH, daytime_loaders[NORMAL_CONDITION], device, epochs=FEATURE_EXTRACTOR_EPOCHS)

    model = build_model(NUM_CLASSES, device, ARCH, SUBCLUSTER_TYPE)
    trainer = AiMotiveDensityTrainer(model, NUM_CLASSES, device)

    print("Training HDC Density Model (daytime only)...")
    trainer.reaccumulate_prototypes(daytime_loaders[NORMAL_CONDITION])
    for epoch in range(1, MAX_HDC_EPOCHS + 1):
        trainer.retrain(daytime_loaders[NORMAL_CONDITION], model, epoch, trainer.logger)

    torch.save(model.state_dict(), HDC_SAVE_PATH)
    print(f"HDC checkpoint saved to {HDC_SAVE_PATH}")

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
            target_dataset = torchdata.ConcatDataset([loader.dataset for loader in adverse_loaders.values()])
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

def incremental_update_test(device: torch.device, ARCH: dict):
    model = build_model(NUM_CLASSES, device, ARCH, SUBCLUSTER_TYPE)
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
        "epoch_accs": [],
    }

    print(f"--- Incremental Evaluation: {UNSUP_EPOCHS} unsupervised epochs per condition ---")

    for cond in ADVERSE_CONDITIONS:
        if cond not in train_loaders:
            continue

        print(f"\n{'='*60}")
        print(f"Unsupervised Update Phase: [{cond.upper()}]")

        val_loader = val_loaders.get(cond) or next(iter(val_loaders.values()))

        acc_pre, _ = test_model(model, val_loader, device)
        print(f"    Epoch 0 (pre) — acc: {acc_pre:.4f}")

        epoch_accs = [acc_pre]

        for ep in range(1, UNSUP_EPOCHS + 1):
            model.train()
            total_updates = 0
            for proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _ in tqdm(train_loaders[cond], desc=f"    epoch {ep}/{UNSUP_EPOCHS} [{cond}]", leave=False):
                if isinstance(proj_in, dict):
                    proj_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proj_in.items()}
                else:
                    proj_in = proj_in.to(device)

                preds = model.inference_update(
                    proj_in,
                    learning_rate=0.01,
                    distance_sensitivity=1.0,
                    thresholds=[0.2, 0.80],
                    beta=0.1,
                )
                total_updates += (preds > 0).sum().item()

            acc_ep, _ = test_model(model, val_loader, device)
            epoch_accs.append(acc_ep)
            print(f"    Epoch {ep} — acc: {acc_ep:.4f}  Δ: {acc_ep - acc_pre:+.4f}  updates: {total_updates}")

        acc_post = epoch_accs[-1]
        print(f"    Final Δ acc: {acc_post - acc_pre:+.4f}")

        history["steps_labels"].append(f"Updates on {cond.capitalize()}")
        history["conditions"].append(cond)
        history["acc_pairs"].append((acc_pre, acc_post))
        history["epoch_accs"].append(epoch_accs)

    save_domain_adaptation_plot(history, file_suffix="_aimotive_condition_split")

def save_domain_adaptation_plot(history: dict, file_suffix: str = ""):
    labels = history["steps_labels"]
    conditions = history["conditions"]
    acc_pairs = np.array(history["acc_pairs"])
    epoch_accs = history.get("epoch_accs", [])

    if len(acc_pairs) == 0:
        print("No data to plot.")
        return

    n = len(labels)
    deltas = acc_pairs[:, 1] - acc_pairs[:, 0]
    has_trajectory = len(epoch_accs) > 0

    fig = plt.figure(figsize=(22, max(8, n * 1.6 + 5)), facecolor="white")
    fig.patch.set_facecolor("white")

    if has_trajectory:
        gs = GridSpec(
            2, 3,
            figure=fig,
            width_ratios=[1.4, 0.8, 1.0],
            height_ratios=[1, 0.15],
            wspace=0.30,
            hspace=0.12,
        )
        ax_dot = fig.add_subplot(gs[0, 0])
        ax_delta = fig.add_subplot(gs[0, 1])
        ax_traj = fig.add_subplot(gs[0, 2])
        ax_leg = fig.add_subplot(gs[1, :])
    else:
        gs = GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[1.6, 1],
            height_ratios=[1, 0.15],
            wspace=0.28,
            hspace=0.12,
        )
        ax_dot = fig.add_subplot(gs[0, 0])
        ax_delta = fig.add_subplot(gs[0, 1])
        ax_traj = None
        ax_leg = fig.add_subplot(gs[1, :])

    C_PRE = "#2171B5"
    C_POST = "#CB181D"
    C_GRID = "#E8E8E8"
    C_TEXT = "#1A1A1A"
    C_SUBTEXT= "#555555"

    axes_to_style = [ax_dot, ax_delta, ax_leg]
    if ax_traj:
        axes_to_style.append(ax_traj)
    for ax in axes_to_style:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
            spine.set_linewidth(0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    y_pos = np.arange(n)

    for yi in range(n):
        cond  = conditions[yi]
        color = CONDITION_COLORS.get(cond, DEFAULT_COLOR)
        ax_dot.axhspan(yi - 0.42, yi + 0.42, color=color, alpha=0.08, zorder=0)
        ax_dot.text(1.01, yi, cond.upper(), transform=ax_dot.get_yaxis_transform(), fontsize=7, color=color, alpha=0.8, va="center", ha="left", fontfamily="monospace")

    ax_dot.hlines(y_pos, acc_pairs[:, 0], acc_pairs[:, 1], color="#AAAAAA", alpha=0.7, linewidth=1.5, zorder=1, linestyle="--")
    ax_dot.scatter(acc_pairs[:, 0], y_pos, color=C_PRE,  s=160, zorder=4, edgecolors="white", linewidths=1.2)
    ax_dot.scatter(acc_pairs[:, 1], y_pos, color=C_POST, s=160, zorder=4, edgecolors="white", linewidths=1.2)

    for yi, (pre, post) in enumerate(acc_pairs):
        ax_dot.text(pre  - 0.003, yi + 0.22, f"{pre:.3f}", fontsize=7.5, color=C_PRE,  ha="right", va="bottom")
        ax_dot.text(post + 0.003, yi + 0.22, f"{post:.3f}", fontsize=7.5, color=C_POST, ha="left",  va="bottom")

    ax_dot.set_yticks(y_pos)
    ax_dot.set_yticklabels(labels, fontsize=9, color=C_TEXT)
    ax_dot.tick_params(axis="x", colors=C_SUBTEXT, labelsize=8)
    ax_dot.tick_params(axis="y", length=0)
    ax_dot.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_dot.grid(axis="x", color=C_GRID, linewidth=0.8, zorder=0)
    ax_dot.set_xlim(
        min(acc_pairs.min() - 0.04, 0.50),
        max(acc_pairs.max() + 0.04, 1.00),
    )
    ax_dot.set_xlabel("Classification accuracy on target domain", fontsize=9, color=C_SUBTEXT, labelpad=8)
    ax_dot.set_title(
        f"Domain Shift: {SOURCE_DOMAIN.capitalize()} → Adverse Conditions\n"
        "Unsupervised Inference-Time Adaptation",
        fontsize=12, color=C_TEXT, fontweight="bold", pad=12, loc="left",
    )
    src_acc = acc_pairs[:, 0].mean()
    ax_dot.axvline(src_acc, color=C_PRE, linewidth=0.8, linestyle=":", alpha=0.5, zorder=2)
    ax_dot.text(src_acc + 0.002, -0.55, f"avg pre\n{src_acc:.3f}", fontsize=6.5, color=C_PRE, alpha=0.7, va="top", ha="left")

    bar_colors = [C_POST if d >= 0 else C_PRE for d in deltas]
    bars = ax_delta.barh(y_pos, deltas, color=bar_colors, alpha=0.75, height=0.55, zorder=2)
    for bar, d in zip(bars, deltas):
        if d < 0:
            bar.set_hatch("///")
            bar.set_edgecolor(C_PRE)
            bar.set_linewidth(0.5)
    ax_delta.axvline(0, color=C_TEXT, linewidth=0.8, zorder=3)
    for yi, d in enumerate(deltas):
        sign = "+" if d >= 0 else ""
        color = C_POST if d >= 0 else C_PRE
        offset = max(abs(deltas).max() * 0.02, 0.0002)
        offset = offset if d >= 0 else -offset
        ax_delta.text(d + offset, yi, f"{sign}{d:.4f}", fontsize=8, color=color, va="center", ha="left" if d >= 0 else "right")
    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels([""] * n)
    ax_delta.tick_params(axis="x", colors=C_SUBTEXT, labelsize=8)
    ax_delta.tick_params(axis="y", length=0)
    ax_delta.grid(axis="x", color=C_GRID, linewidth=0.8, zorder=0)
    ax_delta.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.3f}"))
    ax_delta.set_xlabel("Δ Accuracy after adaptation", fontsize=9, color=C_SUBTEXT, labelpad=8)
    ax_delta.set_title("Accuracy Change", fontsize=11, color=C_TEXT, fontweight="bold", pad=12, loc="left")

    if ax_traj and epoch_accs:
        for ea, cond in zip(epoch_accs, conditions):
            color = CONDITION_COLORS.get(cond, DEFAULT_COLOR)
            epochs_x = np.arange(len(ea))
            ax_traj.plot(epochs_x, ea, color=color, linewidth=2,
                         marker="o", markersize=5,
                         markerfacecolor=color, markeredgecolor="white",
                         label=cond.capitalize(), zorder=3)
            ax_traj.fill_between(epochs_x, ea[0], ea, color=color, alpha=0.08)
            ax_traj.annotate(
                f"{ea[-1]:.3f}",
                xy=(epochs_x[-1], ea[-1]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=color, va="center",
            )

        ax_traj.tick_params(colors=C_SUBTEXT, labelsize=8)
        ax_traj.set_xlabel("Unsupervised epoch", fontsize=9, color=C_SUBTEXT, labelpad=8)
        ax_traj.set_ylabel("Accuracy", fontsize=9, color=C_SUBTEXT, labelpad=8)
        ax_traj.set_title("Adaptation Trajectory", fontsize=11, color=C_TEXT, fontweight="bold", pad=12, loc="left")
        ax_traj.grid(color=C_GRID, linewidth=0.8, zorder=0)
        ax_traj.set_xticks(np.arange(len(epoch_accs[0])))
        ax_traj.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Ep {int(v)}"))
        ax_traj.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
        ax_traj.legend(fontsize=8, frameon=False, labelcolor=C_TEXT, loc="best")

    ax_leg.axis("off")
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PRE, markersize=9, markeredgecolor="white", label=f"Before adaptation (trained on {SOURCE_DOMAIN})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_POST, markersize=9, markeredgecolor="white", label="After unsupervised inference-time adaptation"),
    ]
    cond_patches = [mpatches.Patch(facecolor=CONDITION_COLORS.get(c, DEFAULT_COLOR), alpha=0.6, label=f"Target: {c}") for c in dict.fromkeys(conditions)]
    ax_leg.legend(
        handles=legend_elements + cond_patches,
        loc="center",
        ncol=len(legend_elements) + len(cond_patches),
        fontsize=8.5, frameon=False, labelcolor=C_TEXT,
    )

    note = (
        "Note: all conditions share scene-level class 0 (car) as the dominant label. "
        "Accuracy reflects domain robustness, not class diversity."
    )
    fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=7.5, color=C_SUBTEXT, style="italic")

    fig.suptitle("HDC Model — Unsupervised Domain Adaptation  |  aiMotive LiDAR", fontsize=14, fontweight="bold", color=C_TEXT, y=1.02,)

    plt.tight_layout()
    out = f"domain_adaptation_results{file_suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Domain adaptation plot saved to {out}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset root: {DATA_DIR}")
    print(f"MinEnt: {'enabled' if USE_MENT else 'disabled'}")

    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", "r"))
    except Exception as e:
        print(f"Error opening arch yaml: {e}")
        quit()

    model = pretrain_pipeline(device, ARCH, use_ment=USE_MENT)
    incremental_update_test(device, ARCH)

if __name__ == "__main__":
    main()
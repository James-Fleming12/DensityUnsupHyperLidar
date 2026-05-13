import copy
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from dataset.kitti.parser import Parser
from modules.HDC_utils import Model, DensityModel

from tqdm import tqdm

from unsup_main import train_extractor, train_hdc, init_sub, test_hdc_model

import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from dataset.waymo_data import WaymoDataset
import torch.utils.data as torchdata

MODEL_DIR = "logs"
DATA_DIR = "/mnt/bravo/jmfleming/waymo_skitti"
LOG_DIR = "logs"
NUM_CLASSES = 13

MAX_HDC_EPOCHS = 5
FEATURE_EXTRACTOR_EPOCHS = 30

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

ALL_CONDITIONS = ["sunny", "rain", "fog", "night"]
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != "sunny"]

CONDITION_COLORS = {
    "sunny": "#F5C518",
    "rain":  "#4C9BE8",
    "fog":   "#A0A0A0",
    "night": "#7B4EA0",
}
DEFAULT_COLOR = "#AAAAAA"

def get_loader(ARCH, DATA, sequences, shuffle=True, weather_filter=None):
    """
    Return a Parser initialised in Waymo mode for the given sequences.

    weather_filter : list of condition strings to include, or None for all.
                     e.g. ["sunny"], ["rain", "fog"], None
    """
    return Parser(
        mode="waymo",
        root=DATA_DIR,
        train_sequences=sequences,
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
        shuffle_train=shuffle,
    )

def get_condition_loaders(ARCH, DATA, sequences, batch_size=1, shuffle=False, conditions=None):
    """
    Build one DataLoader per weather condition for the given sequences.

    Returns a dict:  { "sunny": DataLoader, "rain": DataLoader, ... }
    Only conditions that have at least one frame in `sequences` are included.
    """
    if conditions is None:
        conditions = ALL_CONDITIONS

    common_kwargs = dict(
        root=DATA_DIR,
        sequences=sequences,
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        transform=False,
        gt=True,
    )

    loaders = {}
    for cond in conditions:
        ds = WaymoDataset(**common_kwargs, weather_filter=[cond])
        if len(ds) == 0:
            print(f"  [get_condition_loaders] '{cond}' — 0 frames in these sequences, skipping.")
            continue
        loaders[cond] = torchdata.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=ARCH["train"]["workers"],
            drop_last=False,
        )
        print(f"  [get_condition_loaders] '{cond}' — {len(ds)} frames")

    return loaders

def pretrain_pipeline(ARCH, DATA):
    """
    Pretraining uses ALL sunny frames in the dataset so the base model has a clean
    clear-weather foundation before incremental adverse-condition updates.
    """
    print(f"--- Starting Pretraining on ALL sunny scenarios ---")

    PRE_DATA = copy.deepcopy(DATA)
    # Provide a weather filter key in case internal parsers check it
    PRE_DATA["weather_filter"] = ["sunny"]
    
    print("Scanning pretraining sequences for class coverage (sunny frames)...")
    seen_classes = collect_seen_classes(
        ARCH, PRE_DATA, PRE_DATA["split"]["train"], conditions=["sunny"])
    print(f"  Pretraining covers {len(seen_classes)} classes: {sorted(seen_classes)}")

    ARCH["train"]["batch_size"] = 16
    print("Training Feature Extractor (sunny only)...")
    train_extractor(ARCH, PRE_DATA, data_dir=DATA_DIR, epochs=FEATURE_EXTRACTOR_EPOCHS)

    ARCH["train"]["batch_size"] = 2
    print("Training HDC Density Model (sunny only)...")
    model = train_hdc(ARCH, PRE_DATA, data_dir=DATA_DIR, epochs=MAX_HDC_EPOCHS)

    print("Initializing Subclusters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sunny_loaders = get_condition_loaders(
        ARCH, PRE_DATA, PRE_DATA["split"]["train"],
        batch_size=ARCH["train"]["batch_size"],
        shuffle=True,
        conditions=["sunny"])

    if "sunny" in sunny_loaders:
        model.init_subclusters(sunny_loaders["sunny"])
    else:
        raise RuntimeError("No sunny frames found in pretraining sequences.")

    torch.save(model.state_dict(), HDC_SUB_PATH)
    print(f"Pretraining complete. Model saved to {HDC_SUB_PATH}")

    return model, seen_classes

def collect_seen_classes(ARCH, DATA, sequences, max_batches=None, conditions=None):
    """
    Scan frames for observed xentropy class IDs.
    """
    if conditions is not None:
        cond_loaders = get_condition_loaders(
            ARCH, DATA, sequences, batch_size=1, shuffle=False,
            conditions=conditions)
        loaders = list(cond_loaders.values())
    else:
        loaders = [get_loader(ARCH, DATA, sequences, shuffle=False).get_train_set()]

    seen = set()
    for loader in loaders:
        for batch_idx, (_, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) \
                in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            unique_ids = proj_labels.unique().tolist()
            seen.update(int(c) for c in unique_ids if int(c) != 255)
    return seen

def class_ids_to_names(class_ids, DATA):
    inv_map = DATA.get("learning_map_inv", {})
    labels = DATA.get("labels", {})
    names = []
    for cid in sorted(class_ids):
        raw_id = inv_map.get(cid, None)
        name = labels.get(raw_id, f"Class {cid}") if raw_id is not None else f"Class {cid}"
        names.append(name)
    return names

def incremental_update_test(ARCH, DATA, seen_classes=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seqs = DATA["split"]["train"]
    valid_seqs = DATA["split"]["valid"]

    model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device))
    model.to(device)

    print("Building per-condition validation loaders...")
    val_loaders = get_condition_loaders(
        ARCH, DATA, valid_seqs,
        batch_size=1, shuffle=False,
        conditions=ALL_CONDITIONS)

    if not val_loaders:
        raise RuntimeError("No validation frames found for any condition.")

    history = {
        "steps_labels": [],
        "conditions": [],
        "acc_pairs": [],
        "miou_pairs": [],
        "novel_classes": [],
    }

    print(f"--- Incremental Evaluation: Unsupervised Updates on Adverse Conditions ---")
    
    # Load all training frames for the adverse conditions
    train_loaders = get_condition_loaders(
        ARCH, DATA, train_seqs,
        batch_size=1, shuffle=True,
        conditions=ADVERSE_CONDITIONS)

    if not train_loaders:
        print("  No adverse condition frames found in the dataset — skipping.")
        return

    # Iterate through each condition (e.g. rain -> fog -> night)
    for cond in ADVERSE_CONDITIONS:
        if cond not in train_loaders:
            continue

        print(f"\n{'='*60}")
        print(f"Unsupervised Update Phase: [{cond.upper()}]")

        step_label = f"Updates on {cond.capitalize()}"

        novel_in_step = set()
        if seen_classes is not None:
            cond_classes = collect_seen_classes(
                ARCH, DATA, train_seqs, conditions=[cond])
            novel_in_step = cond_classes - seen_classes
            if novel_in_step:
                names = class_ids_to_names(novel_in_step, DATA)
                print(f"    Novel classes: {names}")
            else:
                print("    No novel classes.")

        val_loader_for_cond = val_loaders.get(cond)
        if val_loader_for_cond is None:
            print(f"    No '{cond}' val frames — using full val set.")
            val_loader_for_cond = next(iter(val_loaders.values()))

        acc_pre, miou_pre = test_hdc_model(model, val_loader_for_cond)
        print(f"    Pre  — acc: {acc_pre:.4f}  mIoU: {miou_pre:.4f}")

        model.train()
        for _, (proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _) in \
                enumerate(tqdm(train_loaders[cond], desc=f"    update [{cond}]", leave=False)):
            if proj_in.shape[1] > 0:
                model.inference_update(proj_in.to(device), learning_rate=0.001, distance_sensitivity=3.0)

        acc_post, miou_post = test_hdc_model(model, val_loader_for_cond)
        print(f"    Post — acc: {acc_post:.4f}  mIoU: {miou_post:.4f}  Δ mIoU: {miou_post - miou_pre:+.4f}")

        history["steps_labels"].append(step_label)
        history["conditions"].append(cond)
        history["acc_pairs"].append((acc_pre,  acc_post))
        history["miou_pairs"].append((miou_pre, miou_post))
        history["novel_classes"].append(novel_in_step)

    save_multi_step_dumbbell_ug(history, DATA, file_suffix="_condition_split")

def save_multi_step_dumbbell_ug(history, DATA=None, file_suffix=""):
    labels = history["steps_labels"]
    conditions = history["conditions"]
    acc_pairs = np.array(history["acc_pairs"])
    miou_pairs = np.array(history["miou_pairs"])
    novel_classes = history.get("novel_classes", [set() for _ in labels])

    while len(novel_classes) < len(labels):
        novel_classes.append(set())

    show_table = DATA is not None

    if show_table:
        fig = plt.figure(figsize=(26, max(8, len(labels) * 0.85 + 3)))
        gs  = GridSpec(1, 3, figure=fig, width_ratios=[5, 5, 4], wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(18, max(8, len(labels) * 0.85 + 3)))
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = None

    y_pos = np.arange(len(labels))

    COLOR_PRE = '#4C9BE8'
    COLOR_POST = '#E8574C'
    COLOR_NOVEL_ROW = '#FFF3CD'
    COLOR_NONE_ROW = '#F4F4F4'

    def row_bg(yi):
        has_novel = len(novel_classes[yi]) > 0
        if has_novel:
            return COLOR_NOVEL_ROW
        cond = conditions[yi] if yi < len(conditions) else "sunny"
        base = CONDITION_COLORS.get(cond, DEFAULT_COLOR)
        return base + "33"

    def draw_ax(ax, pairs, title):
        for yi in range(len(pairs)):
            ax.axhspan(yi - 0.45, yi + 0.45, color=row_bg(yi), zorder=0, alpha=0.8)

        ax.hlines(y_pos, pairs[:, 0], pairs[:, 1], color='#AAAAAA', alpha=0.6, linewidth=2, zorder=1)
        ax.scatter(pairs[:, 0], y_pos, color=COLOR_PRE,  s=130, label='Pre-Update',  zorder=3, edgecolors='white', linewidths=0.8)
        ax.scatter(pairs[:, 1], y_pos, color=COLOR_POST, s=130, label='Post-Update', zorder=3, edgecolors='white', linewidths=0.8)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.set_xlabel("Metric Value", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower right', fontsize=9)

    if len(acc_pairs) > 0:
        draw_ax(ax1, acc_pairs, "Accuracy Gain per Condition")
        draw_ax(ax2, miou_pairs, "mIoU Gain per Condition")

        ax1.set_yticks(y_pos)
        tick_labels = ax1.set_yticklabels(labels, fontsize=8)
        for tick, cond in zip(tick_labels, conditions):
            tick.set_color(CONDITION_COLORS.get(cond, "black"))

        ax2.tick_params(labelleft=False)

        cond_patches = [
            mpatches.Patch(color=CONDITION_COLORS[c], label=c.capitalize())
            for c in ALL_CONDITIONS
            if c in conditions
        ]
        ax1.legend(
            handles=cond_patches,
            title="Condition", loc='upper left',
            fontsize=8, title_fontsize=8,
            bbox_to_anchor=(0, -0.06), ncol=len(cond_patches),
            frameon=True, framealpha=0.9,
        )

    if ax3 is not None and len(acc_pairs) > 0:
        ax3.set_xlim(0, 1)
        ax3.set_ylim(len(labels) - 0.5, -0.5)
        ax3.axis('off')
        ax3.set_title("Novel Labels Discovered", fontsize=13, fontweight='bold', pad=10)

        for yi, step_label in enumerate(labels):
            novel = novel_classes[yi]
            has_novel = len(novel) > 0
            row_color = COLOR_NOVEL_ROW if has_novel else COLOR_NONE_ROW
            ax3.axhspan(yi - 0.45, yi + 0.45, color=row_color, alpha=0.7, zorder=0)

            if has_novel:
                names     = class_ids_to_names(novel, DATA)
                cell_text = ", ".join(names)
                if len(cell_text) > 38:
                    cell_text = cell_text[:35] + "..."
                text_color = '#7B4F00'
                marker     = "⚑ "
            else:
                cell_text  = "—"
                text_color = '#888888'
                marker     = ""

            ax3.text(0.05, yi, marker + cell_text, va='center', ha='left', fontsize=8, color=text_color)

        novel_patch = mpatches.Patch(color=COLOR_NOVEL_ROW, alpha=0.7, label='Contains novel labels')
        none_patch  = mpatches.Patch(color=COLOR_NONE_ROW,  alpha=0.7, label='No novel labels')
        ax3.legend(handles=[novel_patch, none_patch], loc='lower center', fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=True, framealpha=0.9)

    plt.suptitle(
        "Impact of Incremental Unsupervised Inference Updates",
        fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = f"incremental_dumbbell_results{file_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dumbbell plot saved to {out_path}")

def save_final_plot(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["miou"], 'r-s', label='mIoU')
    plt.plot(history["steps"], history["acc"],  'b-o', label='Accuracy')
    plt.xlabel('Condition Update Step')
    plt.ylabel('Performance Metrics')
    plt.title('HDC Model Improvement via Incremental Inference Updates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('incremental_update_test.png', dpi=300)
    plt.close()
    print("Plot saved to incremental_update_test.png")

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

    model, seen_classes = pretrain_pipeline(ARCH, DATA)

    incremental_update_test(ARCH, DATA, seen_classes=seen_classes)

if __name__ == "__main__":
    main()
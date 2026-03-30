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

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 20
FEATURE_EXTRACTOR_EPOCHS = 400

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def get_loader(ARCH, DATA, sequences, shuffle=True):
    return Parser(
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
        shuffle_train=shuffle
    )

def pretrain_pipeline(ARCH, DATA, base_count=10):
    """
    Executes the standard training flow on a subset of the data.
 
    Returns
    -------
    model : DensityModel
    seen_classes : set of int
        Mapped class IDs observed in the pretraining sequences.
    """
    print(f"--- Starting Pretraining on first {base_count} scenarios ---")

    PRE_DATA = copy.deepcopy(DATA)
    PRE_DATA["split"]["train"] = DATA["split"]["train"][:base_count]

    print("Scanning pretraining sequences for class coverage...")
    seen_classes = collect_seen_classes(ARCH, PRE_DATA, PRE_DATA["split"]["train"])
    print(f"  Pretraining covers {len(seen_classes)} classes: {sorted(seen_classes)}")

    ARCH["train"]["batch_size"] = 16

    print("Training Feature Extractor...")
    train_extractor(ARCH, PRE_DATA, epochs=150)

    ARCH["train"]["batch_size"] = 2

    print("Training HDC Density Model...")
    model = train_hdc(ARCH, PRE_DATA, epochs=30)

    print("Initializing Subclusters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_loader = get_loader(ARCH, PRE_DATA, PRE_DATA["split"]["train"], shuffle=True).get_train_set()
    model.init_subclusters(base_loader)

    torch.save(model.state_dict(), HDC_SUB_PATH)
    print(f"Pretraining complete. Model saved to {HDC_SUB_PATH}")
 
    return model, seen_classes

def collect_seen_classes(ARCH, DATA, sequences, max_batches=None):
    loader = get_loader(ARCH, DATA, sequences, shuffle=False).get_train_set()
    seen = set()
 
    for batch_idx, (_, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        unique_ids = proj_labels.unique().tolist()
        seen.update(int(c) for c in unique_ids if int(c) != 255)
 
    return seen

def class_ids_to_names(class_ids, DATA):
    inv_map = DATA.get("learning_map_inv", {}) # learning_map_inv maps mapped_id -> one raw label id
    labels = DATA.get("labels", {})
 
    names = []
    for cid in sorted(class_ids):
        raw_id = inv_map.get(cid, None)
        name = labels.get(raw_id, f"Class {cid}") if raw_id is not None else f"Class {cid}"
        names.append(name)
    return names

def incremental_update_test(ARCH, DATA, base_count=10, inc_step=2, seen_classes=None):
    """
    Performs incremental inference updates and tracks Pre vs Post performance.
    If seen_classes is provided, each chunk is scanned for novel class IDs
    not present during pretraining.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    remaining_seqs = DATA["split"]["train"][base_count:]
    valid_seqs = DATA["split"]["valid"]

    model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device))
    model.to(device)

    val_loader = get_loader(ARCH, DATA, valid_seqs, shuffle=False).get_valid_set()

    history = {
        "steps_labels": [],
        "acc_pairs": [],
        "miou_pairs": [],
        "novel_classes": [],
    }

    print(f"--- Starting Incremental Evaluation on {len(remaining_seqs)} scenarios ---")

    for i in range(0, len(remaining_seqs), inc_step):
        chunk = remaining_seqs[i: i + inc_step]
        if not chunk:
            break

        current_range = f"{base_count + i}-{base_count + i + len(chunk)}"
        history["steps_labels"].append(f"Scenarios {current_range}")
        print(f"\nProcessing Batch: {current_range}...")

        novel_in_chunk = set() # --- Novel class detection ---
        if seen_classes is not None:
            print("  Scanning chunk for novel classes...")
            chunk_classes = collect_seen_classes(ARCH, DATA, chunk)
            novel_in_chunk = chunk_classes - seen_classes
            if novel_in_chunk:
                names = class_ids_to_names(novel_in_chunk, DATA)
                print(f"  Novel classes found: {names}")
            else:
                print("  No novel classes in this chunk.")
        history["novel_classes"].append(novel_in_chunk)
 
        acc_pre, miou_pre = test_hdc_model(model, val_loader)
 
        chunk_loader = get_loader(ARCH, DATA, chunk, shuffle=True).get_train_set()
        model.train()
        for _, (proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(tqdm(chunk_loader)):
            if proj_in.shape[1] > 0:
                model.inference_update(proj_in.to(device), learning_rate=0.001, distance_sensitivity=3.0)

        acc_post, miou_post = test_hdc_model(model, val_loader)

        history["acc_pairs"].append((acc_pre, acc_post))
        history["miou_pairs"].append((miou_pre, miou_post))

        print(f"Batch {current_range} Jump: mIoU {miou_pre:.4f} -> {miou_post:.4f}")

    save_multi_step_dumbbell(history, DATA, file_suffix=f"_{base_count}")

def save_final_plot(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["miou"], 'r-s', label='mIoU')
    plt.plot(history["steps"], history["acc"], 'b-o', label='Accuracy')
    plt.xlabel('Training Scenarios Seen')
    plt.ylabel('Performance Metrics')
    plt.title('HDC Model Improvement via Incremental Inference Updates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('incremental_update_test.png', dpi=300)
    plt.close()
    print("Plot saved to incremental_update_test.png")

def save_multi_step_dumbbell(history, DATA=None, file_suffix=""):
    labels = history["steps_labels"]
    acc_pairs = np.array(history["acc_pairs"])
    miou_pairs = np.array(history["miou_pairs"])
    novel_classes = history.get("novel_classes", [set() for _ in labels])

    while len(novel_classes) < len(labels):
        novel_classes.append(set())

    show_table = DATA is not None

    if show_table:
        fig = plt.figure(figsize=(24, max(8, len(labels) * 0.8 + 3)))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[5, 5, 4], wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(16, max(8, len(labels) * 0.8 + 3)))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = None

    y_pos = np.arange(len(labels))

    COLOR_PRE = '#4C9BE8'
    COLOR_POST = '#E8574C'
    COLOR_NOVEL_ROW = '#FFF3CD'
    COLOR_NONE_ROW = '#F4F4F4'

    def draw_ax(ax, pairs, title):
        for yi in range(len(pairs)):
            has_novel = len(novel_classes[yi]) > 0
            ax.axhspan(yi - 0.45, yi + 0.45, color=COLOR_NOVEL_ROW if has_novel else COLOR_NONE_ROW, zorder=0, alpha=0.6)

        ax.hlines(y_pos, pairs[:, 0], pairs[:, 1], color='#AAAAAA', alpha=0.6, linewidth=2, zorder=1)
        ax.scatter(pairs[:, 0], y_pos, color=COLOR_PRE, s=130, label='Pre-Update', zorder=3, edgecolors='white', linewidths=0.8)
        ax.scatter(pairs[:, 1], y_pos, color=COLOR_POST, s=130, label='Post-Update', zorder=3, edgecolors='white', linewidths=0.8)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.set_xlabel("Metric Value", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower right', fontsize=9)

    draw_ax(ax1, acc_pairs,  "Accuracy Gain per Batch")
    draw_ax(ax2, miou_pairs, "mIoU Gain per Batch")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax2.tick_params(labelleft=False)

    if ax3 is not None:
        ax3.set_xlim(0, 1)
        ax3.set_ylim(len(labels) - 0.5, -0.5)
        ax3.axis('off')
        ax3.set_title("Novel Labels in Chunk", fontsize=13, fontweight='bold', pad=10)

        for yi, step_label in enumerate(labels):
            novel = novel_classes[yi]
            has_novel = len(novel) > 0

            row_color = COLOR_NOVEL_ROW if has_novel else COLOR_NONE_ROW
            ax3.axhspan(yi - 0.45, yi + 0.45, color=row_color, alpha=0.7, zorder=0)

            if has_novel:
                names = class_ids_to_names(novel, DATA)
                cell_text = ", ".join(names)
                if len(cell_text) > 38:
                    cell_text = cell_text[:35] + "..."
                text_color = '#7B4F00'
                marker = "⚑ "
            else:
                cell_text = "—"
                text_color = '#888888'
                marker = ""

            ax3.text(0.05, yi, marker + cell_text, va='center', ha='left', fontsize=8, color=text_color)

        novel_patch = mpatches.Patch(color=COLOR_NOVEL_ROW, alpha=0.7, label='Contains novel labels')
        none_patch  = mpatches.Patch(color=COLOR_NONE_ROW,  alpha=0.7, label='No novel labels')
        ax3.legend(handles=[novel_patch, none_patch], loc='lower center', fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=True, framealpha=0.9)

    plt.suptitle("Impact of Incremental Inference Updates on Unseen Data", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = f"incremental_dumbbell_results{file_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dumbbell plot saved to {out_path}")

def main():
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()
 
    base_counts = [4, 6, 8, 10, 12]
    inc_steps = [2]
 
    for base in base_counts:
        model, seen_classes = pretrain_pipeline(ARCH, DATA, base_count=base)
        incremental_update_test(ARCH, DATA, base_count=base, inc_step=inc_steps[0], seen_classes=seen_classes)

if __name__=="__main__":
    main()
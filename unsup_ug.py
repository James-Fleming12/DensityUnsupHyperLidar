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

from unsup_main import convert_dataset

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 20
FEATURE_EXTRACTOR_EPOCHS = 400

SUPERVISED_FRACTION = 0.01

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def select_representative_sequences(ARCH, DATA, n, sequences=None):
    """
    Selects the n sequences from the training split that are most representative
    of the full dataset, using class distribution similarity as the proxy for
    representativeness.

    Strategy: embed each sequence as a class-frequency histogram, then greedily
    pick the subset whose *combined* histogram has the smallest KL divergence
    from the full-dataset histogram.

    Returns a list of n sequence identifiers.
    """
    from scipy.special import kl_div

    all_seqs = sequences or DATA["split"]["train"]

    print(f"  Computing class histograms for {len(all_seqs)} sequences...")
    NUM_MAPPED_CLASSES = NUM_CLASSES

    seq_histograms = {}
    for seq in tqdm(all_seqs, desc="Scanning sequences"):
        loader = get_loader(ARCH, DATA, [seq], shuffle=False).get_train_set()
        counts = np.zeros(NUM_MAPPED_CLASSES, dtype=np.float64)
        for (_, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in loader:
            for cid in proj_labels.unique().tolist():
                cid = int(cid)
                if cid != 255 and cid < NUM_MAPPED_CLASSES:
                    counts[cid] += (proj_labels == cid).sum().item()
        total = counts.sum()
        seq_histograms[seq] = counts / total if total > 0 else counts

    all_counts = np.sum(list(seq_histograms.values()), axis=0)
    all_total = all_counts.sum()
    target_hist = all_counts / all_total if all_total > 0 else all_counts

    EPS = 1e-10

    def symmetric_kl(p, q):
        p, q = p + EPS, q + EPS
        return np.sum(kl_div(p, q) + kl_div(q, p))

    selected  = []
    remaining = list(all_seqs)
    combined  = np.zeros(NUM_MAPPED_CLASSES, dtype=np.float64)

    for step in range(n):
        best_seq,  best_div = None, float('inf')
        for seq in remaining:
            candidate = combined + seq_histograms[seq]
            total     = candidate.sum()
            candidate_norm = candidate / total if total > 0 else candidate
            div = symmetric_kl(candidate_norm, target_hist)
            if div < best_div:
                best_div, best_seq = div, seq

        selected.append(best_seq)
        remaining.remove(best_seq)
        combined += seq_histograms[best_seq]
        print(f"  Step {step+1}/{n}: selected seq '{best_seq}' (KL div: {best_div:.4f})")

    print(f"  Selected {len(selected)} representative sequences: {selected}")
    return selected

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

def pretrain_pipeline(ARCH, DATA, base_count=10, representative=False):
    """
    representative: if True, selects the base_count sequences whose combined class distribution best matches the full dataset, rather than taking the first base_count sequences.
    """
    print(f"--- Starting Pretraining on {base_count} scenarios ({'representative' if representative else 'sequential'}) ---")

    PRE_DATA = copy.deepcopy(DATA)

    if representative:
        print("Selecting representative sequences...")
        selected_seqs = select_representative_sequences(ARCH, DATA, base_count)
        PRE_DATA["split"]["train"] = selected_seqs
    else:
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
 
    val_loader = get_loader(ARCH, PRE_DATA, DATA["split"]["valid"], shuffle=False).get_valid_set()
    acc, miou = test_hdc_model(model, val_loader)
    sup_history = {
        "steps_labels": [f"Scenarios 0-{base_count}"],
        "acc_pairs": [(0.0, acc)],
        "miou_pairs": [(0.0, miou)],
        "novel_classes": [set()],
    }

    return model, seen_classes, sup_history, PRE_DATA["split"]["train"]

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

def incremental_update_test(ARCH, DATA, base_count=10, inc_step=2, seen_classes=None, supervised_seqs=None):
    """
    Performs incremental inference updates and tracks Pre vs Post performance.
    If seen_classes is provided, each chunk is scanned for novel class IDs
    not present during pretraining.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if supervised_seqs is not None:
        supervised_set = set(supervised_seqs)
        remaining_seqs = [s for s in DATA["split"]["train"] if s not in supervised_set]
    else:
        remaining_seqs = DATA["split"]["train"][base_count:]

    valid_seqs = DATA["split"]["valid"]

    model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
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

        current_range = f"{chunk[0]}-{chunk[-1]}" if len(chunk) > 1 else f"{chunk[0]}"
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

    return history

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

def save_multi_step_dumbbell_ug(history, DATA=None, file_suffix=""):
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

def plot_combined_performance(supervised_history, unsupervised_history, file_suffix=""):
    """
    Plots per-pixel accuracy and mIoU across both supervised and unsupervised
    phases on a single graph, with a vertical divider between the two sections.
    """
    sup_labels = supervised_history["steps_labels"]
    unsup_labels = unsupervised_history["steps_labels"]

    sup_acc = [post for _, post in supervised_history["acc_pairs"]]
    sup_miou = [post for _, post in supervised_history["miou_pairs"]]

    unsup_acc = []
    unsup_miou = []
    unsup_x = []
    n_sup = len(sup_labels)

    for i, (pre_a, post_a) in enumerate(unsupervised_history["acc_pairs"]):
        pre_m, post_m = unsupervised_history["miou_pairs"][i]
        base_x = n_sup + i * 2
        unsup_x.extend([base_x, base_x + 1])
        unsup_acc.extend([pre_a, post_a])
        unsup_miou.extend([pre_m, post_m])

    sup_x = list(range(n_sup))

    all_x = sup_x + unsup_x
    all_acc = sup_acc + unsup_acc
    all_miou = sup_miou + unsup_miou

    sup_tick_labels  = [f"S{i+1}\n{l}" for i, l in enumerate(sup_labels)]
    unsup_tick_labels = []
    for i, l in enumerate(unsup_labels):
        base_x = n_sup + i * 2
        unsup_tick_labels.append((base_x, f"U{i+1}pre\n{l}"))
        unsup_tick_labels.append((base_x + 1, f"U{i+1}post\n{l}"))

    fig, ax = plt.subplots(figsize=(max(14, len(all_x) * 0.9 + 4), 6))

    sup_end = n_sup - 0.5
    ax.axvspan(-0.5, sup_end, alpha=0.08, color='steelblue', label='_sup_bg')
    ax.axvspan(sup_end, max(all_x) + 0.5, alpha=0.08, color='darkorange', label='_unsup_bg')

    ax.axvline(x=sup_end, color='black', linewidth=2, linestyle='--', zorder=5)
    ax.text(sup_end - 0.1, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.98, 'Supervised →\n← Unsupervised begins', ha='right', va='top', fontsize=9, color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.plot(sup_x,  sup_acc,  'o-',  color='steelblue',  linewidth=2, markersize=7, label='Accuracy (supervised)')
    ax.plot(sup_x,  sup_miou, 's--', color='steelblue',  linewidth=2, markersize=7, alpha=0.65, label='mIoU (supervised)')

    ax.plot(unsup_x, unsup_acc,  'o-',  color='darkorange', linewidth=2, markersize=7, label='Accuracy (unsupervised)')
    ax.plot(unsup_x, unsup_miou, 's--', color='darkorange', linewidth=2, markersize=7, alpha=0.65, label='mIoU (unsupervised)')

    for i in range(len(unsupervised_history["acc_pairs"])):
        base_x = n_sup + i * 2
        ax.axvspan(base_x - 0.5, base_x + 0.5, alpha=0.06, color='gray')
        ax.annotate('pre',  xy=(base_x,     min(all_acc + all_miou) - 0.01), ha='center', va='top', fontsize=7, color='gray')
        ax.annotate('post', xy=(base_x + 1, min(all_acc + all_miou) - 0.01), ha='center', va='top', fontsize=7, color='gray')

    all_tick_x = sup_x + [x for x, _ in unsup_tick_labels]
    all_tick_labels = sup_tick_labels + [l for _, l in unsup_tick_labels]

    ax.set_xticks(all_tick_x)
    ax.set_xticklabels(all_tick_labels, fontsize=7)
    ax.set_xlim(-0.5, max(all_x) + 0.5)
    ax.set_ylim(max(0, min(all_acc + all_miou) - 0.05), min(1, max(all_acc + all_miou) + 0.05))

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Per-Pixel Accuracy & mIoU: Supervised → Unsupervised", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    y_top = ax.get_ylim()[1]
    ax.text((sup_end - 0.5) / 2, y_top, "SUPERVISED", ha='center', va='bottom', fontsize=11, fontweight='bold', color='steelblue', alpha=0.7)
    ax.text((sup_end + max(all_x) + 0.5) / 2, y_top, "UNSUPERVISED", ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkorange', alpha=0.7)

    plt.tight_layout()
    out_path = f"combined_performance{file_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined performance plot saved to {out_path}")

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

    convert_dataset() # temp thing to deal with badly done data
 
    # base_counts = [4, 6, 8, 10, 12]
    # inc_steps = [2]
 
    # for base in base_counts:
    #     model, seen_classes = pretrain_pipeline(ARCH, DATA, base_count=base)
    #     incremental_update_test(ARCH, DATA, base_count=base, inc_step=inc_steps[0], seen_classes=seen_classes)

    all_seqs = DATA["split"]["train"]
    n_supervised = max(1, int(len(all_seqs) * SUPERVISED_FRACTION))

    print(f"Total training sequences : {len(all_seqs)}")
    print(f"Supervised slice         : {n_supervised}  ({SUPERVISED_FRACTION*100:.1f} %)")
    print(f"Unsupervised slice       : {len(all_seqs) - n_supervised}  ({(1 - SUPERVISED_FRACTION)*100:.1f} %)")

    model, seen_classes, sup_history, supervised_seqs = pretrain_pipeline(ARCH, DATA, base_count=n_supervised, representative=True)

    unsup_history = incremental_update_test(ARCH, DATA, base_count=n_supervised, inc_step=2, seen_classes=seen_classes, supervised_seqs=supervised_seqs)
    
    plot_combined_performance(sup_history, unsup_history, file_suffix=f"_{SUPERVISED_FRACTION}")

if __name__=="__main__":
    main()
import copy
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from dataset.kitti.parser import Parser
from modules.HDC_utils import Model, DensityModel

from tqdm import tqdm

from unsup_main import train_extractor, train_hdc, init_sub, test_hdc_model

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 20
FEATURE_EXTRACTOR_EPOCHS = 400

BASE_COUNT = 4
INC_STEP = 2

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

def pretrain_pipeline(ARCH, DATA, base_count=BASE_COUNT):
    """
    Executes the standard training flow on a subset of the data.
    """
    print(f"--- Starting Pretraining on first {base_count} scenarios ---")

    PRE_DATA = copy.deepcopy(DATA)
    PRE_DATA["split"]["train"] = DATA["split"]["train"][:base_count]

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
    
    return model

def incremental_update_test(ARCH, DATA, base_count=BASE_COUNT, inc_step=INC_STEP):
    """
    Performs incremental inference updates and tracks Pre vs Post performance 
    to create a multi-step dumbbell plot showing the jump at every chunk.
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
        "miou_pairs": []
    }

    print(f"--- Starting Incremental Evaluation on {len(remaining_seqs)} scenarios ---")

    for i in range(0, len(remaining_seqs), inc_step):
        chunk = remaining_seqs[i : i + inc_step]
        if not chunk: break
        
        current_range = f"{base_count + i}-{base_count + i + len(chunk)}"
        history["steps_labels"].append(f"Scenarios {current_range}")
        print(f"\nProcessing Batch: {current_range}...")

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

    save_multi_step_dumbbell(history)

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

def save_multi_step_dumbbell(history):
    """
    Creates a multi-row dumbbell plot showing improvement at every incremental step.
    """
    labels = history["steps_labels"]
    acc_pairs = np.array(history["acc_pairs"])
    miou_pairs = np.array(history["miou_pairs"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    y_pos = np.arange(len(labels))

    def draw_ax(ax, pairs, title, color_pre='#1f77b4', color_post='#d62728'):
        ax.hlines(y_pos, pairs[:, 0], pairs[:, 1], color='grey', alpha=0.5, linewidth=2, zorder=1)
        ax.scatter(pairs[:, 0], y_pos, color=color_pre, s=120, label='Pre-Update', zorder=2)
        ax.scatter(pairs[:, 1], y_pos, color=color_post, s=120, label='Post-Update', zorder=2)
        
        ax.set_title(title, fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_xlabel("Metric Value")
        ax.legend(loc='lower right')

    draw_ax(ax1, acc_pairs, "Accuracy Gain per Batch")
    draw_ax(ax2, miou_pairs, "mIoU Gain per Batch")
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    
    plt.suptitle("Impact of Incremental Inference Updates on Unseen Data", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('incremental_dumbbell_results.png', dpi=300)
    plt.close()
    print("Dumbbell plot saved to incremental_dumbbell_results.png")

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

    _ = pretrain_pipeline(ARCH, DATA)
    incremental_update_test(ARCH, DATA)

if __name__=="__main__":
    main()
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

BASE_COUNT = 10
INC_STEP = 2

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def get_loader(ARCH, DATA, sequences, shuffle=True):
    return Parser(
        root=DATA_DIR,
        train_sequences=sequences,
        valid_sequences=DATA["split"]["valid"],
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    remaining_seqs = DATA["split"]["train"][base_count:]
    valid_seqs = DATA["split"]["valid"]

    model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    val_loader = get_loader(ARCH, DATA, valid_seqs, shuffle=False).get_valid_set()
    
    history = {"steps": [base_count], "acc": [], "miou": []}

    acc, miou = test_hdc_model(model, val_loader)
    history["acc"].append(acc)
    history["miou"].append(miou)
    print(f"Pretrained Baseline -> mIoU: {miou:.4f}")

    for i in range(0, len(remaining_seqs), inc_step):
        chunk = remaining_seqs[i : i + inc_step]
        if not chunk: break
        
        current_total = base_count + i + len(chunk)
        print(f"\n--- Update Step: Total Scenarios {current_total} ---")

        chunk_loader = get_loader(ARCH, DATA, chunk, shuffle=True).get_train_set()
        
        model.train()
        for _, (proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(tqdm(chunk_loader)):
            model.inference_update(
                proj_in.to(device), 
                learning_rate=0.001, 
                distance_sensitivity=3.0
            )

        acc, miou = test_hdc_model(model, val_loader)
        history["acc"].append(acc)
        history["miou"].append(miou)
        history["steps"].append(current_total)
        print(f"Step {current_total} Result -> mIoU: {miou:.4f}")

    save_final_plot(history)

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
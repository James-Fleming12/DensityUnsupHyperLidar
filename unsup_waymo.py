import argparse
import logging
import os
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from unsup_ugw import pretrain_pipeline
from dataset.kitti.parser import Parser

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def save_graphic(save_path, title, data):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    if isinstance(data, dict):
        for label, values in data.items():
            plt.plot(values, label=label)
        plt.legend()
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Metric')
    plt.savefig(save_path)
    plt.close()

def extract_metrics_from_conf_matrix(conf_matrix):
    tp = torch.diag(conf_matrix)
    union = conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - tp
    iou_per_class = tp / (union + 1e-6)
    valid_classes = union > 0 
    miou = iou_per_class[valid_classes].mean().item()
    overall_acc = tp.sum().item() / (conf_matrix.sum().item() + 1e-6)
    return miou, overall_acc

def evaluate_and_adapt(model, target_dataloader, device):
    """Helper method executing the forward/eval/adapt cycle."""
    miou_history = []
    acc_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for _, (proj_in, _, proj_labels, *_) in enumerate(tqdm(target_dataloader, desc="Adapting", leave=False)):
        proj_in = proj_in.to(device)
        proj_labels = proj_labels.to(device).view(-1)
        
        if proj_in.shape[1] > 0:
            # Evaluate: Update global confusion matrix
            model.eval()
            with torch.no_grad():
                logits, sims, indices, _ = model(proj_in)
                predictions = torch.argmax(logits, dim=1)
                selected_labels = proj_labels[indices]
                
                mask = (selected_labels >= 0) & (selected_labels < num_classes)
                if mask.any():
                    hist = torch.bincount(
                        num_classes * selected_labels[mask] + predictions[mask], 
                        minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)
                    cumulative_confusion_matrix += hist
                
            cumulative_miou, cumulative_acc = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
            miou_history.append(cumulative_miou)
            acc_history.append(cumulative_acc)
            
            # Adapt: Inference Update
            model.train()
            model.inference_update(
                proj_in,
                learning_rate=0.001,
                distance_sensitivity=3.0,
                thresholds=[0.45, 0.80]
            )
            
    return {"mIoU": miou_history, "Accuracy": acc_history}

def run_semantic_kitti(model, logger):
    logger.info("Running inference updates on SemanticKITTI...")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
        parser = Parser(root="/mnt/alpha/jmfleming/KITTI", # Sourced from unsup_test.py
                        train_sequences=DATA["split"]["train"],
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
                        shuffle_train=True)
        target_dataloader = parser.get_train_set()
        return evaluate_and_adapt(model, target_dataloader, device)
    except Exception as e:
        logger.error(f"Failed to load SemanticKITTI dataset: {e}")
        return {"mIoU": [], "Accuracy": []}

def run_nuscenes(model, logger):
    logger.info("Running inference updates on NuScenes...")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
        parser = Parser(root="/mnt/alpha/jmfleming/nuscenes_kitti", # Sourced from unsup_main.py
                        train_sequences=DATA["split"]["train"],
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
                        shuffle_train=True)
        target_dataloader = parser.get_train_set()
        return evaluate_and_adapt(model, target_dataloader, device)
    except Exception as e:
        logger.error(f"Failed to load NuScenes dataset: {e}")
        return {"mIoU": [], "Accuracy": []}

def load_hdc_model(path):
    print(f"Loading pretrained HDC model from {path}...")
    from modules.HDC_utils import DensityModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    NUM_CLASSES = 13
    model = DensityModel(ARCH, "logs", 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {path}, starting from random init.")
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Pretrain HDC on Waymo and Test Inference on KITTI/NuScenes")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model on Waymo instead of loading')
    parser.add_argument('--pretrained_path', type=str, default='logs/hdc_sub.pth', help='Path to load pretrained Waymo model')
    parser.add_argument('--log_dir', type=str, default='logs/waymo_pretrain_test', help='Directory to save logs and graphics')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'waymo_test.log')
    logger = setup_logger(log_file)
    
    try:
        WAYMO_ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
        WAYMO_DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
        if args.pretrain:
            logger.info("Starting Waymo Pretraining...")
            model, trainer = pretrain_pipeline(WAYMO_ARCH, WAYMO_DATA, return_trainer=True)
            
            # Save the feature extractor optimizer so the user can resume or extend training later
            opt_path = os.path.join(args.log_dir, 'feature_optimizer.pth')
            torch.save(trainer.optimizer.state_dict(), opt_path)
            logger.info(f"Successfully pretrained model on Waymo. Optimizer state saved to {opt_path}")
        else:
            logger.info(f"Loading pretrained Waymo model from {args.pretrained_path}...")
            model = load_hdc_model(args.pretrained_path)
    except Exception as e:
        logger.error(f"Failed to pretrain model: {e}")
        return

    # Test SemanticKITTI
    kitti_data = run_semantic_kitti(model, logger)
    if kitti_data["mIoU"]:
        save_graphic(os.path.join(args.log_dir, 'waymo_to_kitti.png'), 'Waymo -> SemanticKITTI', kitti_data)

    # Test NuScenes
    nuscenes_data = run_nuscenes(model, logger)
    if nuscenes_data["mIoU"]:
        save_graphic(os.path.join(args.log_dir, 'waymo_to_nuscenes.png'), 'Waymo -> NuScenes', nuscenes_data)

    logger.info("Completed Waymo Inference Tests!")

if __name__ == "__main__":
    main()

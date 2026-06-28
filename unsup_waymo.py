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

def save_improvement_bar_chart(save_path, title, data):
    import numpy as np
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    metrics = list(data.keys())
    initial_vals = [data[m][0] if len(data[m]) > 0 else 0 for m in metrics]
    final_vals = [data[m][-1] if len(data[m]) > 0 else 0 for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, initial_vals, width, label='Initial (Zero-Shot)', color='#4C9BE8')
    plt.bar(x + width/2, final_vals, width, label='Final (Adapted)', color='#E8574C')
    
    plt.ylabel('Metric Value')
    plt.title(f'{title} - Total Improvement')
    plt.xticks(x, metrics)
    plt.legend()
    
    for i, v in enumerate(initial_vals):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(final_vals):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0, max(max(initial_vals), max(final_vals) if len(final_vals) > 0 else 0) * 1.2 + 0.05)
    plt.tight_layout()
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

    for _, batch_data in enumerate(tqdm(target_dataloader, desc="Adapting", leave=False)):
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
        proj_xyz = batch_data[10].to(device) if len(batch_data) > 10 else None
        
        if proj_in.shape[1] > 0:
            # Evaluate: Update global confusion matrix
            model.eval()
            with torch.no_grad():
                # D3CTTA returns h as the 4th element
                logits, sims, indices, h = model(proj_in)
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
            if hasattr(model, 'G_d'):  # Duck typing for D3CTTA
                model.inference_update(
                    h=h,
                    predictions=predictions,
                    xyz=proj_xyz
                )
            else:
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
    modeldir = os.path.dirname(path)
    model = DensityModel(ARCH, modeldir, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def load_d3ctta_model(path):
    print(f"Loading pretrained feature extractor for D3CTTA from {path}...")
    from modules.network.ResNet import ResNet_34
    from modules.D3CTTA import D3CTTA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 13
    feature_extractor = ResNet_34(NUM_CLASSES, aux=False, use_adaptor=True)
    
    # Try to load the best extractor from SENet_valid_best
    se_path = os.path.join(os.path.dirname(path), "SENet_valid_best")
    if os.path.exists(se_path):
        w_dict = torch.load(se_path, map_location=device)
        feature_extractor.load_state_dict(w_dict['state_dict'], strict=False)
        print(f"Loaded feature extractor weights from {se_path}")
    else:
        print(f"Warning: SENet_valid_best not found at {se_path}, starting from random init.")
    
    feature_extractor.to(device)
    feature_extractor.eval()
    
    model = D3CTTA(feature_extractor, num_classes=NUM_CLASSES)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Pretrain HDC on Waymo and Test Inference on KITTI/NuScenes")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model on Waymo instead of loading')
    parser.add_argument('--pretrained_path', type=str, default='logs/hdc_sub.pth', help='Path to load pretrained Waymo model')
    parser.add_argument('--log_dir', type=str, default='logs/waymo_pretrain_test', help='Directory to save logs and graphics')
    parser.add_argument('--compare', action='store_true', help='Use D3CTTA with pretrained feature extractor instead of HDC')
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
            
            if args.compare:
                logger.info("Compare flag enabled. Loading D3CTTA model...")
                model = load_d3ctta_model(args.pretrained_path)
        else:
            if args.compare:
                logger.info(f"Loading pretrained feature extractor for D3CTTA from {args.pretrained_path}...")
                model = load_d3ctta_model(args.pretrained_path)
            else:
                logger.info(f"Loading pretrained Waymo model from {args.pretrained_path}...")
                model = load_hdc_model(args.pretrained_path)
    except Exception as e:
        logger.error(f"Failed to load or pretrain model: {e}")
        return

    suffix = "_d3ctta" if args.compare else ""
    # Test SemanticKITTI
    kitti_data = run_semantic_kitti(model, logger)
    if kitti_data["mIoU"]:
        save_graphic(os.path.join(args.log_dir, f'waymo_to_kitti{suffix}.png'), 'Waymo -> SemanticKITTI', kitti_data)
        save_improvement_bar_chart(os.path.join(args.log_dir, f'waymo_to_kitti_bar{suffix}.png'), 'Waymo -> SemanticKITTI', kitti_data)

    # Test NuScenes
    nuscenes_data = run_nuscenes(model, logger)
    if nuscenes_data["mIoU"]:
        save_graphic(os.path.join(args.log_dir, f'waymo_to_nuscenes{suffix}.png'), 'Waymo -> NuScenes', nuscenes_data)
        save_improvement_bar_chart(os.path.join(args.log_dir, f'waymo_to_nuscenes_bar{suffix}.png'), 'Waymo -> NuScenes', nuscenes_data)

    logger.info("Completed Waymo Inference Tests!")

if __name__ == "__main__":
    main()

import argparse
import logging
import os
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from unsup_ugw import pretrain_pipeline, get_condition_loaders, ADVERSE_CONDITIONS
from modules.HDC_utils import DensityModel
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

def pretrain_hdc_model():
    print("Pretraining HDC model...")
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    model = pretrain_pipeline(ARCH, DATA)
    return model

def load_hdc_model(path):
    print(f"Loading pretrained HDC model from {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    NUM_CLASSES = 13 # Default Waymo classes from unsup_ugw.py
    
    model = DensityModel(ARCH, "logs", 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {path}, starting from random init.")
    model.to(device)
    return model

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
            # 5. Evaluate: Update global confusion matrix
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
            
            # 6. Adapt
            model.train()
            model.inference_update(
                proj_in,
                learning_rate=0.001,
                distance_sensitivity=3.0,
                thresholds=[0.45, 0.80]
            )
            
    return {"mIoU": miou_history, "Accuracy": acc_history}

def run_intra_weather(model, logger):
    logger.info("Running Intra-Dataset UDA (Weather) experiment: Waymo (Sunny) -> Waymo (Rain/Night)")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    target_loaders = get_condition_loaders(ARCH, DATA, DATA["split"]["train"], batch_size=1, shuffle=True, conditions=ADVERSE_CONDITIONS)
    
    all_metrics = {"mIoU": [], "Accuracy": []}
    for cond in ADVERSE_CONDITIONS:
        if cond in target_loaders:
            logger.info(f"Adapting on {cond} condition...")
            metrics = evaluate_and_adapt(model, target_loaders[cond], device)
            all_metrics["mIoU"].extend(metrics["mIoU"])
            all_metrics["Accuracy"].extend(metrics["Accuracy"])
            
    return all_metrics

def run_inter_geography(model, logger):
    logger.info("Running Inter-Dataset UDA (Geography) experiment: Waymo -> SemanticKITTI")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
        parser = Parser(root="/mnt/alpha/jmfleming/semantic_kitti", # Path guess based on unsup_main structure
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

def run_inter_density(model, logger):
    logger.info("Running Inter-Dataset UDA (Density) experiment: Waymo -> NuScenes")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
        parser = Parser(root="/mnt/alpha/jmfleming/nuscenes_kitti",
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

def run_ada(model, logger):
    logger.info("Running Active Domain Adaptation (ADA) experiment...")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    target_loaders = get_condition_loaders(ARCH, DATA, DATA["split"]["train"], batch_size=1, shuffle=True, conditions=["rain"])
    
    if "rain" in target_loaders:
        # TODO: Inject ADA Oracle budgeting filters directly into evaluate_and_adapt loop here
        return evaluate_and_adapt(model, target_loaders["rain"], device)
    return {"mIoU": [], "Accuracy": []}

def run_online_stability(model, logger):
    logger.info("Running Online Stability over Time experiment...")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    target_loaders = get_condition_loaders(ARCH, DATA, DATA["split"]["train"], batch_size=1, shuffle=False, conditions=["rain"])
    
    if "rain" in target_loaders:
        return evaluate_and_adapt(model, target_loaders["rain"], device)
    return {"mIoU": [], "Accuracy": []}

def run_compute_efficiency(model, logger):
    logger.info("Running Compute & Efficiency Cost experiment...")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    efficiency_history = []
    memory_history = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    target_loaders = get_condition_loaders(ARCH, DATA, DATA["split"]["train"], batch_size=1, shuffle=True, conditions=["rain"])
    if not target_loaders or "rain" not in target_loaders:
        return {"Adaptation Time (ms)": [], "Peak GPU Memory (MB)": []}
        
    target_dataloader = target_loaders["rain"]
    for _, (proj_in, _, proj_labels, *_) in enumerate(tqdm(target_dataloader, desc="Efficiency Tracking", leave=False)):
        proj_in = proj_in.to(device)
        if proj_in.shape[1] == 0: continue
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        if torch.cuda.is_available():
            start_event.record()
            
        model.train()
        model.inference_update(
            proj_in,
            learning_rate=0.001,
            distance_sensitivity=3.0,
            thresholds=[0.45, 0.80]
        )
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            frame_cost_ms = start_event.elapsed_time(end_event)
            frame_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            logger.error("WARNING: CUDA is not available! Cannot track async GPU compute time or VRAM.")
            raise RuntimeError("CUDA must be available to run Compute & Efficiency Cost Tracking without dummy data.")
            
        efficiency_history.append(frame_cost_ms)
        memory_history.append(frame_mem_mb)

    return {"Adaptation Time (ms)": efficiency_history, "Peak GPU Memory (MB)": memory_history}

def main():
    parser = argparse.ArgumentParser(description="HDC Baseline Unsupervised & Active Domain Adaptation Experiments")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the HDC model inside the script instead of loading')
    parser.add_argument('--pretrained_path', type=str, default='logs/hdc_sub.pth', help='Path to load pretrained HDC model')
    parser.add_argument('--log_dir', type=str, default='logs/hdc_baseline', help='Directory to save in-depth logs and graphics')
    
    # Experiment flags
    parser.add_argument('--intra_weather', action='store_true', help='Run Intra-Dataset UDA (Weather)')
    parser.add_argument('--inter_geography', action='store_true', help='Run Inter-Dataset UDA (Geography)')
    parser.add_argument('--inter_density', action='store_true', help='Run Inter-Dataset UDA (Density)')
    parser.add_argument('--ada', action='store_true', help='Run Active Domain Adaptation (ADA)')
    parser.add_argument('--online_stability', action='store_true', help='Run Online Stability over Time (Ablation)')
    parser.add_argument('--compute_efficiency', action='store_true', help='Run Compute & Efficiency Cost (Ablation)')
    
    args = parser.parse_args()

    # Pretrain or load HDC model
    if args.pretrain:
        model = pretrain_hdc_model()
    else:
        model = load_hdc_model(args.pretrained_path)

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'experiment_run.log')
    logger = setup_logger(log_file)
    logger.info("Started HDC Baseline Experiments")

    if args.intra_weather:
        data = run_intra_weather(model, logger)
        save_graphic(os.path.join(args.log_dir, 'intra_weather.png'), 'Intra-Dataset UDA (Weather)', data)
    
    if args.inter_geography:
        data = run_inter_geography(model, logger)
        save_graphic(os.path.join(args.log_dir, 'inter_geography.png'), 'Inter-Dataset UDA (Geography)', data)
        
    if args.inter_density:
        data = run_inter_density(model, logger)
        save_graphic(os.path.join(args.log_dir, 'inter_density.png'), 'Inter-Dataset UDA (Density)', data)
        
    if args.ada:
        data = run_ada(model, logger)
        save_graphic(os.path.join(args.log_dir, 'ada.png'), 'Active Domain Adaptation (ADA)', data)
        
    if args.online_stability:
        data = run_online_stability(model, logger)
        save_graphic(os.path.join(args.log_dir, 'online_stability.png'), 'Online Stability over Time', data)
        
    if args.compute_efficiency:
        data = run_compute_efficiency(model, logger)
        save_graphic(os.path.join(args.log_dir, 'compute_efficiency.png'), 'Compute & Efficiency Cost', data)

if __name__ == "__main__":
    main()

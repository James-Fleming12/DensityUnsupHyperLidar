import argparse
import logging
import os
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from common.laserscan import SemLaserScan, LaserScan
from dataset.kitti.parser import Parser
from unsup_main import train_extractor, train_hdc
from unsup_waymo import extract_metrics_from_conf_matrix, setup_logger, save_graphic, load_hdc_model

def corrupt_beam(points, severity):
    distances = np.linalg.norm(points[:, :3], axis=1)
    pitch = np.arcsin(points[:, 2] / (distances + 1e-6))
    bins = np.linspace(np.min(pitch), np.max(pitch), 65)
    ring_ids = np.digitize(pitch, bins)
    drop_fraction = 0.05 * severity 
    unique_rings = np.unique(ring_ids)
    num_drop = int(len(unique_rings) * drop_fraction)
    dropped_rings = np.random.choice(unique_rings, num_drop, replace=False)
    mask = ~np.isin(ring_ids, dropped_rings)
    return points[mask]

def corrupt_crosstalk(points, severity):
    num_points = len(points)
    noise_fraction = 0.02 * severity 
    num_noise = int(num_points * noise_fraction)
    min_bounds = np.min(points[:, :3], axis=0)
    max_bounds = np.max(points[:, :3], axis=0)
    noise_xyz = np.random.uniform(min_bounds, max_bounds, size=(num_noise, 3))
    noise_intensity = np.random.uniform(0, 0.1, size=(num_noise, 1)) 
    noise_points = np.hstack((noise_xyz, noise_intensity))
    return np.vstack((points, noise_points))

def corrupt_fog(points, severity):
    distances = np.linalg.norm(points[:, :3], axis=1)
    beta = 0.005 * severity 
    survival_prob = np.exp(-beta * distances)
    random_draw = np.random.uniform(0, 1, size=len(points))
    mask = random_draw < survival_prob
    return points[mask]

def corrupt_echo(points, severity):
    intensity_threshold = np.percentile(points[:, 3], 90)
    high_ref_mask = points[:, 3] > intensity_threshold
    echo_points = points[high_ref_mask].copy()
    shift_multiplier = 1.0 + (0.1 * severity) 
    echo_points[:, :3] = echo_points[:, :3] * shift_multiplier
    echo_points[:, 3] = echo_points[:, 3] * 0.5 
    return np.vstack((points, echo_points))

def corrupt_motion(points, severity):
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    timeline = (azimuth - np.min(azimuth)) / (np.max(azimuth) - np.min(azimuth) + 1e-6)
    max_translation = 0.2 * severity 
    blur_shift = np.outer(timeline, np.array([max_translation, 0, 0])) 
    points[:, :3] += blur_shift
    return points

def corrupt_snow(points, severity):
    num_flakes = 1000 * severity
    flake_xyz = np.random.uniform(-10, 10, size=(num_flakes, 3)) 
    flake_intensity = np.random.uniform(0.5, 1.0, size=(num_flakes, 1)) 
    snowflakes = np.hstack((flake_xyz, flake_intensity))
    ground_mask = points[:, 2] < -1.0 
    drop_prob = 0.1 * severity
    survive_ground = np.random.uniform(0, 1, size=np.sum(ground_mask)) > drop_prob
    final_points_mask = np.ones(len(points), dtype=bool)
    final_points_mask[ground_mask] = survive_ground
    points = points[final_points_mask]
    return np.vstack((points, snowflakes))

def apply_corruption(points, corruption_type, severity):
    if corruption_type == 'beam':
        return corrupt_beam(points, severity)
    elif corruption_type == 'crosstalk':
        return corrupt_crosstalk(points, severity)
    elif corruption_type == 'fog':
        return corrupt_fog(points, severity)
    elif corruption_type == 'echo':
        return corrupt_echo(points, severity)
    elif corruption_type == 'motion':
        return corrupt_motion(points, severity)
    elif corruption_type == 'snow':
        return corrupt_snow(points, severity)
    return points

class LiDARCorruptionWrapper(Dataset):
    def __init__(self, base_dataset, corruption_type=None, severity=1):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        original_open = SemLaserScan.open_scan
        original_laser_open = LaserScan.open_scan
        
        wrapper_self = self
        
        def patched_open_scan(scan_self, filename):
            scan_self.reset()
            scan = np.fromfile(filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            
            if wrapper_self.corruption_type:
                scan = apply_corruption(scan, wrapper_self.corruption_type, wrapper_self.severity)
                
            points = scan[:, 0:3]
            remissions = scan[:, 3]
            
            if scan_self.drop_points is not False:
                scan_self.points_to_drop = np.random.randint(0, len(points)-1, int(len(points)*scan_self.drop_points))
                points = np.delete(points, scan_self.points_to_drop, axis=0)
                remissions = np.delete(remissions, scan_self.points_to_drop)

            scan_self.set_points(points, remissions)

        SemLaserScan.open_scan = patched_open_scan
        LaserScan.open_scan = patched_open_scan
        
        try:
            data = self.base_dataset[idx]
        finally:
            SemLaserScan.open_scan = original_open
            LaserScan.open_scan = original_laser_open
            
        return data

def evaluate_and_adapt(model, target_dataloader, device):
    miou_history = []
    acc_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for proj_in, _, proj_labels, *_ in tqdm(target_dataloader, desc="Adapting", leave=False):
        proj_in = proj_in.to(device)
        proj_labels = proj_labels.to(device).view(-1)
        
        if proj_in.shape[1] > 0:
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
            
            model.train()
            model.inference_update(
                proj_in,
                learning_rate=0.001,
                distance_sensitivity=3.0,
                thresholds=[0.45, 0.80]
            )
            
    return {"mIoU": miou_history, "Accuracy": acc_history}

def pretrain_pipeline(ARCH, DATA, data_dir, pretrained_path, return_trainer=False):
    import unsup_main
    log_base = os.path.dirname(pretrained_path)
    os.makedirs(log_base, exist_ok=True)
    
    unsup_main.LOG_DIR = log_base
    unsup_main.MODEL_DIR = log_base
    unsup_main.HDC_SAVE_PATH = os.path.join(log_base, "hdc.pth")
    unsup_main.HDC_SUB_PATH = pretrained_path

    print(f"Pretraining feature extractor on {data_dir}...")
    trainer = train_extractor(ARCH, DATA, data_dir=data_dir, return_trainer=True)
    
    print(f"Pretraining HDC density model on {data_dir}...")
    model, _ = train_hdc(ARCH, DATA, data_dir=data_dir, return_extractor=True)
    
    print("Initializing subclusters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = Parser(root=data_dir,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=True)
    
    dataloader = parser.get_train_set()
    model.init_subclusters(dataloader)
    
    torch.save(model.state_dict(), pretrained_path)
    print(f"Subcluster Initialized Model saved to {pretrained_path}")
    
    if return_trainer:
        return model, trainer
    return model

def save_degradation_plot(save_path, title, data_dict, metric="mIoU"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    severities = [1, 2, 3, 4, 5]
    for corr, sev_dict in data_dict.items():
        vals = [sev_dict.get(s, 0) for s in severities]
        plt.plot(severities, vals, marker='o', label=corr)
    
    plt.title(f"{title} - {metric} Degradation")
    plt.xlabel("Severity")
    plt.ylabel(metric)
    plt.xticks(severities)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on NuScenes-C")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model on standard NuScenes')
    parser.add_argument('--pretrained_path', type=str, default='logs/nusc_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/nusc_c_test', help='Directory to save logs and graphics')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, 'nusc_c.log'))

    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return

    data_dir = "/mnt/alpha/jmfleming/nuscenes_kitti"

    if args.pretrain:
        logger.info(f"Starting Pretraining on standard NuScenes at {data_dir}...")
        model, trainer = pretrain_pipeline(ARCH, DATA, data_dir=data_dir, pretrained_path=args.pretrained_path, return_trainer=True)
        
        opt_path = os.path.join(os.path.dirname(args.pretrained_path), 'feature_optimizer.pth')
        torch.save(trainer.optimizer.state_dict(), opt_path)
        logger.info(f"Successfully pretrained model on NuScenes. Optimizer state saved to {opt_path}")
    else:
        logger.info(f"Loading pretrained model from {args.pretrained_path}")
        model = load_hdc_model(args.pretrained_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    corruptions = ['beam', 'crosstalk', 'fog', 'echo', 'motion', 'snow']
    severities = [1, 2, 3, 4, 5]

    results_miou = {c: {} for c in corruptions}
    results_acc = {c: {} for c in corruptions}
    
    for ctype in corruptions:
        for sev in severities:
            logger.info(f"Testing {ctype} severity {sev}")
            
            parser_obj = Parser(root=data_dir,
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
                            shuffle_train=False)
            
            target_dataset = parser_obj.validloader.dataset
            corrupted_dataset = LiDARCorruptionWrapper(target_dataset, corruption_type=ctype, severity=sev)
            target_dataloader = DataLoader(corrupted_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            model = load_hdc_model(args.pretrained_path)
            
            metrics = evaluate_and_adapt(model, target_dataloader, device)
            
            if len(metrics["mIoU"]) > 0:
                final_miou = metrics["mIoU"][-1]
                final_acc = metrics["Accuracy"][-1]
                
                results_miou[ctype][sev] = final_miou
                results_acc[ctype][sev] = final_acc
                
                logger.info(f"Result for {ctype}-{sev}: mIoU={final_miou:.4f}, Acc={final_acc:.4f}")
                
                save_graphic(os.path.join(args.log_dir, f'traj_{ctype}_{sev}.png'), f'{ctype} Sev {sev}', metrics)
            else:
                logger.info(f"No valid frames evaluated for {ctype}-{sev}")

    save_degradation_plot(os.path.join(args.log_dir, 'degradation_miou.png'), 'NuScenes-C', results_miou, metric='mIoU')
    save_degradation_plot(os.path.join(args.log_dir, 'degradation_acc.png'), 'NuScenes-C', results_acc, metric='Accuracy')
    
    with open(os.path.join(args.log_dir, 'results.json'), 'w') as f:
        json.dump({'mIoU': results_miou, 'Accuracy': results_acc}, f, indent=4)

if __name__ == "__main__":
    main()

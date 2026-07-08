import torch
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.aug_model import AugModel
from dataset.kitti.parser import Parser
from unsup_waymo import extract_metrics_from_conf_matrix
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA_NUSC = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    
    pretrained_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = AugModel(ARCH, os.path.dirname(pretrained_path), 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    model.eval()

    nusc_sensor = ARCH["dataset"]["sensor"].copy()
    nusc_sensor["fov_up"] = 10.0
    nusc_sensor["fov_down"] = -30.0
    nusc_sensor["img_prop"] = nusc_sensor["img_prop"].copy()
    nusc_sensor["img_prop"]["height"] = 32
    nusc_sensor["img_prop"]["width"] = 1024

    parser_obj = Parser(root="/mnt/alpha/jmfleming/nuscenes_kitti",
                        train_sequences=[854],
                        valid_sequences=[854],
                        test_sequences=None,
                        labels=DATA_NUSC["labels"],
                        color_map=DATA_NUSC.get("color_map", {}),
                        learning_map=DATA_NUSC["learning_map"],
                        learning_map_inv=DATA_NUSC["learning_map_inv"],
                        sensor=nusc_sensor,
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=1,
                        workers=1,
                        gt=True,
                        shuffle_train=False)
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False)
    
    num_classes = 17
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    print("Evaluating 20 frames of NuScenes to get per-class IoU...")
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= 20:
            break
            
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
        
        with torch.no_grad():
            logits, sims, indices, h = model(proj_in)
            predictions = torch.argmax(logits, dim=1)
            selected_labels = proj_labels[indices]
            
            mask = (selected_labels > 0) & (selected_labels < num_classes) # Ignore class 0
            if mask.any():
                hist = torch.bincount(
                    num_classes * selected_labels[mask] + predictions[mask], 
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)
                cumulative_confusion_matrix += hist

    _, _, iou_per_class = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
    
    inv_map = DATA_NUSC["learning_map_inv"]
    kitti_labels = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["labels"]
    
    print("\n" + "="*50)
    print("PER-CLASS IoU ON NUSCENES (FROZEN MODEL)")
    print("="*50)
    for class_idx in range(1, 17):
        # The class index maps to a specific original KITTI label via learning_map_inv
        orig_label = inv_map.get(class_idx, 0)
        class_name = kitti_labels.get(orig_label, f"Class {class_idx}")
        iou = iou_per_class[class_idx] * 100
        
        # Check if this class even exists in the confusion matrix (was it in the ground truth?)
        total_gt = cumulative_confusion_matrix[class_idx].sum().item()
        if total_gt == 0:
            print(f"{class_name.ljust(20)}: N/A (0 points in GT)")
        else:
            print(f"{class_name.ljust(20)}: {iou:.2f}%")

if __name__ == "__main__":
    main()

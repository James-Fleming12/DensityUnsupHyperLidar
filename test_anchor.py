import os
import yaml
import torch
import json
from torch.utils.data import DataLoader
from dataset.kitti.parser import Parser
import importlib

unsup_kitti_c = importlib.import_module("unsup_kitti-c")
LiDARCorruptionWrapper = unsup_kitti_c.LiDARCorruptionWrapper
evaluate_and_adapt = unsup_kitti_c.evaluate_and_adapt
load_hdc_model = unsup_kitti_c.load_hdc_model
save_graphic = unsup_kitti_c.save_graphic

MODEL_DIR = "logs/kitti_pretrain"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
PRETRAINED_PATH = os.path.join(MODEL_DIR, "hdc_sub.pth")
CONFIG_PATH = "config/arch/senet-2048p.yml"
LABELS_PATH = "config/labels/semantic-kitti-all.yaml"
SAVE_DIR = "logs/diagnostics"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ARCH = yaml.safe_load(open(CONFIG_PATH, 'r'))
    DATA = yaml.safe_load(open(LABELS_PATH, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    corruptions = ['snow', 'fog', 'motion']
    severities = [2]
    
    methods_to_test = ['frozen', 'exp_a_anchor_off', 'exp_a_anchor_on']
    
    for method in methods_to_test:
        print(f"\n=========================================")
        print(f"Testing Method: {method}")
        print(f"=========================================")
        
        for ctype in corruptions:
            for sev in severities:
                print(f"  -> Condition: {ctype} sev {sev}")
                
                parser_obj = Parser(
                    root=DATA_DIR, train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"], test_sequences=None,
                    labels=DATA["labels"], color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
                    batch_size=1, workers=0, gt=True, shuffle_train=False
                )
                
                target_dataset = parser_obj.validloader.dataset
                corrupted_dataset = LiDARCorruptionWrapper(target_dataset, corruption_type=ctype, severity=sev)
                target_dataloader = DataLoader(corrupted_dataset, batch_size=1, shuffle=False, num_workers=0)
                
                model = load_hdc_model(PRETRAINED_PATH)
                
                eval_only = (method == 'frozen')
                metrics = evaluate_and_adapt(model, target_dataloader, device, eval_only=eval_only, update_method=method)
                
                if len(metrics["mIoU"]) > 0:
                    traj_json_path = os.path.join(SAVE_DIR, f'traj_{ctype}_{sev}_{method}.json')
                    with open(traj_json_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                        
                    save_graphic(os.path.join(SAVE_DIR, f'traj_{ctype}_{sev}_{method}.png'), f'{ctype} Sev {sev} ({method})', metrics)
                    print(f"      Initial mIoU: {metrics['mIoU'][0]:.4f} -> Final mIoU: {metrics['mIoU'][-1]:.4f}")
                else:
                    print(f"      No valid frames evaluated!")

if __name__ == "__main__":
    main()

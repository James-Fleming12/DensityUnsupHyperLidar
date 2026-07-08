import os
import sys
import yaml
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset.kitti.parser import Parser

def check_density():
    CONFIG_ARCH = "config/arch/senet-2048p.yml"
    CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    DATA = yaml.safe_load(open(CONFIG_LABELS, 'r'))

    print("\n" + "="*50)
    print("TESTING PROJECTION DENSITY: KITTI vs NuScenes")
    print("="*50)

    # KITTI
    try:
        kitti_parser = Parser(root="/mnt/alpha/jmfleming/KITTI",
                              train_sequences=[8], valid_sequences=[8], test_sequences=None,
                              labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                              learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
                              sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
                              batch_size=1, workers=1, gt=False, shuffle_train=False)
        
        kitti_data = kitti_parser.validloader.dataset[0] # Get first frame
        kitti_proj = kitti_data[0] # [5, 64, 2048]
        kitti_mask = kitti_data[1] # [64, 2048]
        
        kitti_density = (kitti_mask > 0).float().mean().item() * 100
        print(f"KITTI Projection Density: {kitti_density:.2f}% of pixels contain data.")
    except Exception as e:
        print(f"KITTI test failed: {e}")

    # NuScenes
    try:
        nusc_sensor = ARCH["dataset"]["sensor"].copy()
        nusc_sensor["fov_up"] = 10.0
        nusc_sensor["fov_down"] = -30.0
        nusc_parser = Parser(root="/mnt/alpha/jmfleming/nuscenes_kitti",
                              train_sequences=[854], valid_sequences=[854], test_sequences=None,
                              labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                              learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
                              sensor=nusc_sensor, max_points=ARCH["dataset"]["max_points"],
                              batch_size=1, workers=1, gt=False, shuffle_train=False)
        
        nusc_data = nusc_parser.validloader.dataset[0] # Get first frame
        nusc_proj = nusc_data[0] # [5, 64, 2048]
        nusc_mask = nusc_data[1] # [64, 2048]
        
        nusc_density = (nusc_mask > 0).float().mean().item() * 100
        print(f"NuScenes Projection Density: {nusc_density:.2f}% of pixels contain data.")
        
        if nusc_density < 35.0:
            print("\n>>> CRITICAL FINDING: The NuScenes projection is EXTREMELY SPARSE!")
            print(">>> The neural network was trained on KITTI images which are ~85-95% dense.")
            print(">>> If it is fed an image where >65% of the pixels are empty (-1),")
            print(">>> the 2D Convolution kernels will be multiplying over mostly zeros,")
            print(">>> causing complete network collapse (0.35% mIoU) regardless of the remission fix.")
            print(">>> FIX: We must lower the resolution (e.g. 32x1024) for NuScenes!")
    except Exception as e:
        print(f"NuScenes test failed: {e}")

if __name__ == "__main__":
    check_density()

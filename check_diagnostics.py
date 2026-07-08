import os
import sys
import yaml
import torch
import numpy as np
import glob
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.aug_model import AugModel
from dataset.kitti.parser import Parser

def check_nuscenes_channels():
    print("="*50)
    print("CHECKING NUSCENES DATASET SHAPE (Geometric Mangling Test)")
    print("="*50)
    files = glob.glob('/mnt/alpha/jmfleming/nuscenes_kitti/sequences/*/velodyne/*.bin')
    if not files:
        print("Could not find NuScenes bin files to test.")
        return
        
    f = files[0]
    scan = np.fromfile(f, dtype=np.float32)
    print(f"File: {f}")
    print(f"Total floats in file: {scan.shape[0]}")
    
    div_4 = (scan.shape[0] % 4 == 0)
    div_5 = (scan.shape[0] % 5 == 0)
    
    print(f"Divisible by 4 (KITTI standard): {div_4}")
    print(f"Divisible by 5 (NuScenes standard): {div_5}")
    
    if div_5 and not div_4:
        print("\n>>> CRITICAL FINDING: The NuScenes .bin files have 5 channels (x, y, z, intensity, ring)!")
        print(">>> Because the KITTI parser reads `.reshape(-1, 4)`, it is mathematically mangling the coordinates!")
        print(">>> Row 1: x, y, z, int")
        print(">>> Row 2: ring, x, y, z")
        print(">>> The model is seeing a completely unrecognizable fun-house mirror. This is why mIoU is 0.35%!\n")

def check_kitti_c_firerates():
    print("="*50)
    print("CHECKING KITTI-C HDC FIRING RATES")
    print("="*50)
    
    CONFIG_ARCH = "config/arch/senet-2048p.yml"
    CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
    PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    DATA = yaml.safe_load(open(CONFIG_LABELS, 'r'))
    
    print(f"Loading pretrained HDC model from {PRETRAINED}...")
    model = AugModel(ARCH, os.path.dirname(PRETRAINED), 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(PRETRAINED, map_location=device))
    model.to(device)
    model.eval()
    
    # Load just one corruption chunk to test
    corruption_root = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/fog/heavy"
    print(f"Loading sample from {corruption_root}...")
    
    parser_obj = Parser(root=corruption_root,
                        train_sequences=DATA["split"]["valid"],
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA.get("color_map", {}),
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=1,
                        workers=2,
                        gt=True,
                        shuffle_train=False)
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False)
    
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= 5: # Just test 5 frames
            break
            
        proj_in = batch_data[0].to(device)
        proj_xyz = batch_data[10].to(device) if len(batch_data) > 10 else None
        
        # Test exp_a_anchor_off
        model._firing_log = []
        model.inference_update_soft_consensus(
            proj_in,
            learning_rate=0.001,
            use_consensus_gate=True,
            use_volume_weight=True,
            use_subcluster_gate=True,
            use_anchor=False,
            proj_xyz=proj_xyz
        )
        
        firerate = model._firing_log[0] * 100 if len(model._firing_log) > 0 else 0
        print(f"Frame {batch_idx}: exp_a_anchor_off Firing Rate = {firerate:.2f}%")
        
    print("\n>>> If Firing Rate is 0.00%, the gating mechanism (consensus or subclusters) is too strict!")
    print(">>> This perfectly explains why the initial and final mIoUs were identically matching.")

if __name__ == "__main__":
    check_nuscenes_channels()
    check_kitti_c_firerates()

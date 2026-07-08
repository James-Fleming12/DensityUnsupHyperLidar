import torch
import os

path = "logs/kitti_pretrain/hdc_sub.pth"
ckpt = torch.load(path, map_location='cpu')

subclusters = ckpt.get("subclusters", None)

if subclusters is not None:
    print(f"subclusters shape: {subclusters.shape}")
    print(f"subclusters mean: {subclusters.mean().item():.6f}")
    print(f"subclusters max: {subclusters.max().item():.6f}")
    print(f"subclusters min: {subclusters.min().item():.6f}")
    print(f"subclusters fraction of exact zeros: {(subclusters == 0).float().mean().item() * 100:.2f}%")
else:
    print("subclusters not found in checkpoint!")
    
prototypes = ckpt.get("classify.weight", None)
if prototypes is not None:
    print(f"\nprototypes shape: {prototypes.shape}")
    print(f"prototypes fraction of exact zeros: {(prototypes == 0).float().mean().item() * 100:.2f}%")

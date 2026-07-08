import numpy as np
import glob

print("Checking KITTI axes...")
kitti_files = glob.glob('/mnt/alpha/jmfleming/KITTI/sequences/08/velodyne/*.bin')
if kitti_files:
    scan = np.fromfile(kitti_files[0], dtype=np.float32).reshape(-1, 4)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = scan[:, i]
        print(f"KITTI {axis} Min: {vals.min():.2f}, Max: {vals.max():.2f}, Mean: {vals.mean():.2f}")
    
print("\nChecking NuScenes axes...")
nusc_files = glob.glob('/mnt/alpha/jmfleming/nuscenes_kitti/sequences/0854/velodyne/*.bin')
if nusc_files:
    scan = np.fromfile(nusc_files[0], dtype=np.float32).reshape(-1, 4)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = scan[:, i]
        print(f"NuScenes {axis} Min: {vals.min():.2f}, Max: {vals.max():.2f}, Mean: {vals.mean():.2f}")

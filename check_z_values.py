import numpy as np
import glob

print("Checking KITTI Z values...")
kitti_files = glob.glob('/mnt/alpha/jmfleming/KITTI/sequences/08/velodyne/*.bin')
if kitti_files:
    scan = np.fromfile(kitti_files[0], dtype=np.float32).reshape(-1, 4)
    z_vals = scan[:, 2]
    print(f"KITTI Z Min: {z_vals.min():.2f}, Max: {z_vals.max():.2f}, Mean: {z_vals.mean():.2f}")
    
print("\nChecking NuScenes Z values...")
nusc_files = glob.glob('/mnt/alpha/jmfleming/nuscenes_kitti/sequences/0854/velodyne/*.bin')
if nusc_files:
    scan = np.fromfile(nusc_files[0], dtype=np.float32).reshape(-1, 4)
    z_vals = scan[:, 2]
    print(f"NuScenes Z Min: {z_vals.min():.2f}, Max: {z_vals.max():.2f}, Mean: {z_vals.mean():.2f}")

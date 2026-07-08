import numpy as np
import glob

print("Checking NuScenes intensities...")
files = glob.glob('/mnt/alpha/jmfleming/nuscenes_kitti/sequences/0854/velodyne/*.bin')
if files:
    scan = np.fromfile(files[0], dtype=np.float32).reshape(-1, 4)
    remissions = scan[:, 3]
    print(f"NuScenes Remission Min: {remissions.min()}, Max: {remissions.max()}, Mean: {remissions.mean()}")
    
print("Checking KITTI intensities...")
files = glob.glob('/mnt/alpha/jmfleming/KITTI/sequences/08/velodyne/*.bin')
if files:
    scan = np.fromfile(files[0], dtype=np.float32).reshape(-1, 4)
    remissions = scan[:, 3]
    print(f"KITTI Remission Min: {remissions.min()}, Max: {remissions.max()}, Mean: {remissions.mean()}")

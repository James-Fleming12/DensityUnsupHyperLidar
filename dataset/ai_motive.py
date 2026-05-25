import os
import glob
import json
import numpy as np
import torch
import laspy
from torch.utils.data import Dataset

CLASS_MAP = {
    "car": 0,
    "van": 0,
    "pickup": 0,
    "size_vehicle_m": 0,

    "truck": 1,
    "bus": 1,
    "truck/bus": 1,
    "train": 1,
    "trailer": 1,
    "size_vehicle_xl": 1,

    "motorcycle": 2,
    "bicycle": 2,
    "bike": 2,

    "pedestrian": 3,
    "person": 3,

    "traffic_cone": 4,
    "barrier": 4,
    "misc": 4
}
NUM_CLASSES = len(CLASS_MAP)

ALL_CONDITIONS = ["highway", "urban", "night", "rain"]
NORMAL_CONDITION = "highway"
ADVERSE_CONDITIONS = ["night", "rain"]

PC_RANGE = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]
BEV_SHAPE = (512, 512)
VOXEL_SIZE = [(PC_RANGE[3] - PC_RANGE[0]) / BEV_SHAPE[1], (PC_RANGE[4] - PC_RANGE[1]) / BEV_SHAPE[0], PC_RANGE[5] - PC_RANGE[2]]
MAX_POINTS_PER_VOXEL = 32
MAX_VOXELS = 10000

class AiMotiveDataset(Dataset):
    def __init__(self, root, split="train", conditions=None, val_fraction=0.2):
        self.root = root
        self.split = split
        self.conditions = conditions or ["highway", "night", "rain", "urban"]
        self.frames = []

        for cond in self.conditions:
            cond_path = os.path.join(root, split, cond)
            if not os.path.exists(cond_path):
                continue
                
            sequences = sorted(os.listdir(cond_path))

            split_idx = int(len(sequences) * (1 - val_fraction))
            target_seqs = sequences[:split_idx] if split == "train" else sequences[split_idx:]
            
            for seq in target_seqs:
                seq_path = os.path.join(cond_path, seq)

                lidar_dir = os.path.join(seq_path, "dynamic", "raw-revolutions")
                label_dir = os.path.join(seq_path, "dynamic", "box", "3d_body")
                
                if os.path.exists(lidar_dir):
                    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.laz")))
                    for lf in lidar_files:
                        json_name = os.path.basename(lf).replace('.laz', '.json')
                        self.frames.append({"lidar_path": lf, "label_path": os.path.join(label_dir, json_name), "condition": cond})

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        las = laspy.read(frame["lidar_path"])

        intensity = np.array(las.intensity, dtype=np.float32)
        if intensity.max() > 255.0:
            intensity = (intensity / 65535.0) * 255.0
            
        points = np.vstack((las.x, las.y, las.z, intensity)).transpose()

        labels = []
        if os.path.exists(frame["label_path"]):
            with open(frame["label_path"], 'r') as f:
                data = json.load(f)
                
            objects = data.get("CapturedObjects", []) 
            for obj in objects:
                actor_name = obj.get("ActorName", "").lower()
                cls_name = actor_name.split(" ")[0] if actor_name else ""
                
                if cls_name in CLASS_MAP:
                    labels.append(CLASS_MAP[cls_name])

        if len(labels) == 0:
            labels = [-1]
            
        return points, np.array(labels, dtype=np.int64)

def voxelize(points):
    """
    Groups (N, 4) points into pillars for the PointPillarEncoder.
    Returns: voxel_features (M, 32, 4), voxel_coords (M, 3)
    """
    mask = (
        (points[:, 0] >= PC_RANGE[0]) & (points[:, 0] <= PC_RANGE[3]) &
        (points[:, 1] >= PC_RANGE[1]) & (points[:, 1] <= PC_RANGE[4]) &
        (points[:, 2] >= PC_RANGE[2]) & (points[:, 2] <= PC_RANGE[5])
    )
    points = points[mask]

    voxel_coords = np.floor(
        (points[:, [0, 1, 2]] - np.array([PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]])) / np.array(VOXEL_SIZE)
    ).astype(np.int32)

    voxel_coords = voxel_coords[:, [2, 1, 0]] 
    
    unique_coords, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    
    num_voxels = min(len(unique_coords), MAX_VOXELS)
    
    voxel_features = np.zeros((num_voxels, MAX_POINTS_PER_VOXEL, 4), dtype=np.float32)
    final_coords = np.zeros((num_voxels, 3), dtype=np.int32)
    
    # 4. Populate voxels
    voxel_point_counts = np.zeros(num_voxels, dtype=np.int32)
    for i, voxel_idx in enumerate(inverse_indices):
        if voxel_idx >= num_voxels:
            continue
            
        count = voxel_point_counts[voxel_idx]
        if count < MAX_POINTS_PER_VOXEL:
            voxel_features[voxel_idx, count, :] = points[i]
            final_coords[voxel_idx, :] = unique_coords[voxel_idx]
            voxel_point_counts[voxel_idx] += 1

    return voxel_features, final_coords

def _parser_collate(batch):
    """
    Takes batch of (points, labels), applies voxelization, and 
    formats them for PointPillarEncoder.
    """
    batched_voxel_features = []
    batched_voxel_coords = []
    batched_labels = []
    
    for batch_idx, (points, labels) in enumerate(batch):
        v_feats, v_coords = voxelize(points)

        v_feats_tensor = torch.tensor(v_feats, dtype=torch.float32)
        v_coords_tensor = torch.tensor(v_coords, dtype=torch.long)

        batch_idx_tensor = torch.full((v_coords_tensor.shape[0], 1), batch_idx, dtype=torch.long)
        coords_with_batch = torch.cat([batch_idx_tensor, v_coords_tensor], dim=1)
        
        batched_voxel_features.append(v_feats_tensor)
        batched_voxel_coords.append(coords_with_batch)
        batched_labels.append(torch.tensor(labels, dtype=torch.long))

    final_voxel_features = torch.cat(batched_voxel_features, dim=0)
    final_voxel_coords = torch.cat(batched_voxel_coords, dim=0)

    proj_in = {
        "voxel_features": final_voxel_features,
        "voxel_coords": final_voxel_coords,
        "batch_size": len(batch)
    }

    proj_labels = torch.cat(batched_labels, dim=0)

    return proj_in, None, proj_labels, None, None, None, None, None, None, None, None, None, None, None, None
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as torchdata

CLASS_MAP: Dict[str, int] = {
    "Car":          0,
    "Truck":        1,
    "Bus":          2,
    "Motorcycle":   3,
    "Bicycle":      4,
    "Pedestrian":   5,
    "Other":        6,
}

ALL_CONDITIONS   = ["daytime", "night", "rainy"]
NORMAL_CONDITION = "daytime"
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != NORMAL_CONDITION]

NUM_CLASSES = len(CLASS_MAP)

def read_pcd(path: str) -> np.ndarray:
    """
    Read a PCD file and return an (N, 4) float32 array [x, y, z, intensity].

    Handles both ASCII and binary_compressed PCD formats.
    Falls back to zeros on parse failure so the pipeline never crashes
    on a single corrupt frame.
    """
    try:
        with open(path, "rb") as f:
            header = {}
            header_lines = 0
            for raw_line in f:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                header_lines += 1
                if line.startswith("DATA"):
                    data_fmt = line.split()[1]
                    break
                parts = line.split()
                if len(parts) >= 2:
                    header[parts[0]] = parts[1:]

            n_pts = int(header.get("POINTS", ["0"])[0])
            fields = header.get("FIELDS", ["x", "y", "z"])
            sizes  = [int(s) for s in header.get("SIZE", ["4"] * len(fields))]

            if data_fmt == "ascii":
                pts = []
                for raw_line in f:
                    vals = raw_line.decode("utf-8", errors="ignore").strip().split()
                    if vals:
                        pts.append([float(v) for v in vals])
                arr = np.array(pts, dtype=np.float32)
            elif data_fmt == "binary":
                raw = f.read()
                dtype_list = []
                for field, size in zip(fields, sizes):
                    dtype_list.append((field, np.float32 if size == 4 else np.float64))
                arr_struct = np.frombuffer(raw[:n_pts * sum(sizes)],
                                           dtype=np.dtype(dtype_list))
                arr = np.column_stack(
                    [arr_struct[f].astype(np.float32) for f in fields]
                )
            else:
                return np.zeros((1, 4), dtype=np.float32)

        if arr.shape[1] >= 4:
            return arr[:, :4].astype(np.float32)
        elif arr.shape[1] == 3:
            intensity = np.zeros((arr.shape[0], 1), dtype=np.float32)
            return np.concatenate([arr[:, :3], intensity], axis=1)
        else:
            return np.zeros((1, 4), dtype=np.float32)

    except Exception:
        return np.zeros((1, 4), dtype=np.float32)

def voxelise(points: np.ndarray, voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4.0), point_cloud_range: List[float] = [-51.2, -51.2, -3.0, 51.2, 51.2, 1.0], max_voxels: int = 20000, max_pts_per_voxel: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple numpy voxelisation — pillars style.

    Returns
    -------
    voxel_features : (V, max_pts_per_voxel, 4)  float32
    voxel_coords   : (V, 4)                      int32  [0, z, y, x]
    """
    pc_min = np.array(point_cloud_range[:3], dtype=np.float32)
    pc_max = np.array(point_cloud_range[3:], dtype=np.float32)
    vsize = np.array(voxel_size, dtype=np.float32)

    mask = np.all((points[:, :3] >= pc_min) & (points[:, :3] < pc_max), axis=1)
    points = points[mask]

    if len(points) == 0:
        vf = np.zeros((1, max_pts_per_voxel, 4), dtype=np.float32)
        vc = np.zeros((1, 4), dtype=np.int32)
        return vf, vc

    grid_idx = np.floor((points[:, :3] - pc_min) / vsize).astype(np.int32)
    grid_size = np.ceil((pc_max - pc_min) / vsize).astype(np.int32)

    keys = (grid_idx[:, 0] * grid_size[1] * grid_size[2] + grid_idx[:, 1] * grid_size[2] + grid_idx[:, 2])

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    if len(unique_keys) > max_voxels:
        unique_keys = unique_keys[:max_voxels]
        keep = inverse < max_voxels
        points  = points[keep]
        inverse = inverse[keep]

    n_vox = len(unique_keys)
    vf = np.zeros((n_vox, max_pts_per_voxel, 4), dtype=np.float32)
    cnt = np.zeros(n_vox, dtype=np.int32)

    for pi, vi in enumerate(inverse):
        if vi < n_vox and cnt[vi] < max_pts_per_voxel:
            vf[vi, cnt[vi]] = points[pi, :4]
            cnt[vi] += 1

    x_idx = unique_keys // (grid_size[1] * grid_size[2])
    y_idx = (unique_keys % (grid_size[1] * grid_size[2])) // grid_size[2]
    z_idx = unique_keys % grid_size[2]
    vc = np.stack([np.zeros(n_vox, dtype=np.int32), z_idx, y_idx, x_idx], axis=1).astype(np.int32)

    return vf, vc

def load_annotation(json_path: str, class_map: Dict[str, int]) -> int:
    """
    Read a frame_XXXXXXX.json annotation file and return the integer class ID
    of the most frequent object class in the frame.

    Falls back to 0 (Car) if the file is missing or empty.
    """
    if not os.path.isfile(json_path):
        return 0

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return 0

    objects = data.get("objects", [])
    if not objects:
        return 0

    counts: Dict[int, int] = {}
    for obj in objects:
        cls_str = obj.get("class", "Other")
        cls_id  = class_map.get(cls_str, class_map.get("Other", 0))
        counts[cls_id] = counts.get(cls_id, 0) + 1

    return max(counts, key=counts.__getitem__)

class AiMotiveDataset(torchdata.Dataset):
    """
    PyTorch Dataset for the aiMotive Multimodal Dataset.

    Each sample is one LiDAR keyframe, voxelised into pillars and paired
    with the dominant object class label from the 3D bounding box annotations.

    Parameters
    ----------
    root : path to the dataset root (contains daytime/, night/, rainy/)
    split : 'train' | 'val'  (uses an 80/20 sequence split by default)
    conditions : list of condition strings to include, e.g. ['daytime']
    class_map : dict mapping class string → int; defaults to CLASS_MAP above
    voxel_cfg : dict with keys voxel_size, point_cloud_range, max_voxels, max_pts_per_voxel  (all optional, defaults match PointPillars)
    val_fraction : fraction of sequences held out for validation (default 0.2)
    seed : random seed for the train/val sequence split
    """

    def __init__(self, root: str, split: str = "train", conditions: Optional[List[str]] = None, class_map: Optional[Dict[str, int]] = None, voxel_cfg: Optional[dict] = None, val_fraction: float = 0.2, seed: int = 42,):
        super().__init__()
        self.root = root
        self.split = split
        self.conditions = conditions or ALL_CONDITIONS
        self.class_map = class_map or CLASS_MAP
        self.num_classes = max(self.class_map.values()) + 1

        vcfg = voxel_cfg or {}
        self.voxel_size = vcfg.get("voxel_size", (0.16, 0.16, 4.0))
        self.point_cloud_range = vcfg.get("point_cloud_range", [-51.2, -51.2, -3.0, 51.2, 51.2, 1.0])
        self.max_voxels = vcfg.get("max_voxels", 20000)
        self.max_pts = vcfg.get("max_pts_per_voxel", 32)

        self.val_fraction = val_fraction
        self.seed = seed

        self.samples: List[dict] = self._index_samples()

    def _index_samples(self) -> List[dict]:
        """
        Walk the directory tree and build a list of
        {lidar_path, annotation_path, condition, frame_id} dicts.
        Then split into train / val by sequence so no sequence
        straddles both sets.
        """
        all_sequences: List[Tuple[str, str]] = []

        for cond in self.conditions:
            cond_dir = os.path.join(self.root, cond)
            if not os.path.isdir(cond_dir):
                continue
            for seq_id in sorted(os.listdir(cond_dir)):
                lidar_dir = os.path.join(cond_dir, seq_id, "sensor", "lidar")
                if os.path.isdir(lidar_dir):
                    all_sequences.append((cond, seq_id))

        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(all_sequences))
        n_val   = max(1, int(len(all_sequences) * self.val_fraction))

        if self.split == "val":
            chosen = set(indices[:n_val].tolist())
        else:
            chosen = set(indices[n_val:].tolist())

        selected_seqs = [all_sequences[i] for i in range(len(all_sequences)) if i in chosen]

        samples = []
        for cond, seq_id in selected_seqs:
            lidar_dir = os.path.join(self.root, cond, seq_id, "sensor", "lidar")
            ann_dir = os.path.join(self.root, cond, seq_id, "annotation", "3d_body")

            for fname in sorted(os.listdir(lidar_dir)):
                if not (fname.endswith(".pcd") or fname.endswith(".bin")):
                    continue

                stem = os.path.splitext(fname)[0]
                parts = stem.rsplit("_", 1)
                frame_num = parts[-1] if len(parts) == 2 else "0000001"

                ann_path = os.path.join(ann_dir, f"frame_{frame_num}.json")

                samples.append({
                    "lidar":     os.path.join(lidar_dir, fname),
                    "annotation": ann_path,
                    "condition": cond,
                    "frame_id":  frame_num,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        voxel_batch : dict
            'voxel_features' (V, max_pts, 4)  float32
            'voxel_coords' (V, 4) int32   [0, z, y, x]
            'batch_size' 1
        label : torch.LongTensor  shape ()  - dominant class in frame
        condition : str
        """
        s = self.samples[idx]

        points = read_pcd(s["lidar"])
        vf, vc = voxelise(
            points,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_voxels=self.max_voxels,
            max_pts_per_voxel=self.max_pts,
        )

        voxel_batch = {
            "voxel_features": torch.from_numpy(vf),
            "voxel_coords": torch.from_numpy(vc),
            "batch_size": 1,
        }

        label = load_annotation(s["annotation"], self.class_map)
        return voxel_batch, torch.tensor(label, dtype=torch.long), s["condition"]

def aimotive_collate(batch):
    """
    Collate a list of (voxel_dict, label, condition) samples into a
    batch, correctly setting batch_idx in voxel_coords.
    """
    vf_list, vc_list, labels, conditions = [], [], [], []

    for b_idx, (vdict, lbl, cond) in enumerate(batch):
        vf = vdict["voxel_features"]
        vc = vdict["voxel_coords"].clone()
        vc[:, 0] = b_idx
        vf_list.append(vf)
        vc_list.append(vc)
        labels.append(lbl)
        conditions.append(cond)

    voxel_batch = {
        "voxel_features": torch.cat(vf_list, dim=0),
        "voxel_coords": torch.cat(vc_list, dim=0),
        "batch_size": len(batch),
    }
    return voxel_batch, torch.stack(labels), conditions

class AiMotiveParser:
    """
    Thin wrapper around AiMotiveDataset that mirrors the interface of the
    KITTI/Waymo Parser used by DensityTrainer, so DensityTrainer can be
    reused without modification.

    DensityTrainer calls:
        parser.get_train_set() -> DataLoader  (15-tuple per batch)
        parser.get_valid_set() -> DataLoader
        parser.get_n_classes() -> int

    The DataLoader batches here yield 15-tuples to match exactly:
        (proj_in, proj_mask, proj_labels, unproj_labels,
         path_seq, path_name, p_x, p_y,
         proj_range, unproj_range,
         _, _, _, _, npoints)

    For aiMotive:
        proj_in     = voxel_batch dict  (passed straight to DensityModel.encode)
        proj_mask   = None placeholder
        proj_labels = (B,) integer class labels
        everything else = None placeholder
    """

    def __init__(self, root: str, conditions: Optional[List[str]] = None, batch_size: int = 4, workers: int = 4, val_fraction: float = 0.2, class_map: Optional[Dict[str, int]] = None, voxel_cfg: Optional[dict] = None, seed: int = 42):
        self.root = root
        self.conditions = conditions or ALL_CONDITIONS
        self.batch_size = batch_size
        self.workers = workers
        self.class_map = class_map or CLASS_MAP
        self.num_classes = max(self.class_map.values()) + 1
        self.voxel_cfg = voxel_cfg or {}
        self.val_fraction = val_fraction
        self.seed = seed

    def get_n_classes(self)-> int:
        return self.num_classes

    def _make_loader(self, split: str, conditions: List[str], shuffle: bool) -> torchdata.DataLoader:
        ds = AiMotiveDataset(
            root=self.root,
            split=split,
            conditions=conditions,
            class_map=self.class_map,
            voxel_cfg=self.voxel_cfg,
            val_fraction=self.val_fraction,
            seed=self.seed,
        )
        return torchdata.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=_parser_collate,
            drop_last=False,
        )

    def get_train_set(self) -> torchdata.DataLoader:
        return self._make_loader("train", self.conditions, shuffle=True)

    def get_valid_set(self) -> torchdata.DataLoader:
        return self._make_loader("val", self.conditions, shuffle=False)

def _parser_collate(batch):
    """
    Produces the 15-tuple format expected by DensityTrainer.train() /
    retrain() / validate().

    Tuple layout:
      0  proj_in - voxel dict
      1  proj_mask - None
      2  proj_labels - (B,) LongTensor
      3  unproj_labels - None
      4  path_seq - list of condition strings (plays role of path_seq)
      5  path_name - list of empty strings
      6  p_x - None
      7  p_y - None
      8  proj_range - None
      9  unproj_range - None
      10-13 - None
      14 npoints - None
    """
    voxel_batch, labels, conditions = aimotive_collate(batch)
    nones = [None] * 9
    return (voxel_batch, None, labels, None, conditions, [""] * len(conditions), None, None, None, None, None, None, None, None, None)
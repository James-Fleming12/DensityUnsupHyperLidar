import os
import random
import numpy as np
import torch
from tqdm import tqdm
import glob
from torch.utils.data import Dataset

import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as dp
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection

from common.laserscan import SemLaserScan, LaserScan

WAYMO_TO_SKITTI_LABEL = {
    # Waymo ID : SemanticKITTI ID
    0:  0,   # TYPE_UNDEFINED            → unlabeled
    1:  9,   # TYPE_CAR                  → car
    2:  10,  # TYPE_TRUCK                → truck
    3:  18,  # TYPE_BUS                  → bus (mapped to 'other-vehicle')
    4:  18,  # TYPE_OTHER_VEHICLE        → other-vehicle
    5:  16,  # TYPE_MOTORCYCLIST         → motorcyclist
    6:  15,  # TYPE_BICYCLIST            → bicyclist
    7:  13,  # TYPE_PEDESTRIAN           → person
    8:  0,   # TYPE_SIGN                 → unlabeled (no SKITTI equivalent)
    9:  17,  # TYPE_TRAFFIC_LIGHT        → traffic-sign (closest)
    10: 0,   # TYPE_POLE                 → unlabeled
    11: 0,   # TYPE_CONSTRUCTION_CONE    → unlabeled
    12: 15,  # TYPE_BICYCLE              → bicycle
    13: 16,  # TYPE_MOTORCYCLE           → motorcycle
    14: 0,   # TYPE_BUILDING             → unlabeled (not in SKITTI moving classes)
    15: 72,  # TYPE_VEGETATION           → vegetation
    16: 70,  # TYPE_TREE_TRUNK           → vegetation
    17: 40,  # TYPE_CURB                 → road (closest flat surface)
    18: 40,  # TYPE_ROAD                 → road
    19: 44,  # TYPE_LANE_MARKER          → parking
    20: 48,  # TYPE_OTHER_GROUND         → other-ground
    21: 51,  # TYPE_WALKABLE             → sidewalk
    22: 52,  # TYPE_SIDEWALK             → sidewalk
}

def _weather_tag(stats) -> str:
    """
    Map Waymo segment-level stats to a canonical condition string.
    """
    weather = stats.weather.lower() if hasattr(stats, "weather") else "sunny"
    tod    = stats.time_of_day.lower() if hasattr(stats, "time_of_day") else "day"
 
    if "rain" in weather:
        return "rain"
    if "fog" in weather:
        return "fog"
    if "night" in tod:
        return "night"
    if "dawn" in tod or "dusk" in tod:
        return "night"   # treat dawn/dusk as night for curriculum purposes
    return "sunny"

def _is_scan(filename: str) -> bool:
    return filename.endswith(".bin")

def _is_label(filename: str) -> bool:
    return filename.endswith(".label")

class WaymoConverter:
    """
    Convert Waymo Open Dataset TFRecords to SemanticKITTI layout.
 
    Args:
        tfrecord_dir   : Root directory containing *.tfrecord files.
                         Searched recursively.
        output_dir     : Where to write the sequences/ folder tree.
        scene_id_offset: Integer added to the Waymo segment index to avoid
                         collisions when merging with nuScenes-KITTI sequences.
                         Default 500 keeps well clear of KITTI (0-21) and
                         nuScenes (0-850 padded to 4 digits).
        top_lidar_only : If True, use only the TOP lidar (matches the single-
                         lidar assumption of the rest of this codebase).
    """
    def __init__(self, tfrecord_dir: str, output_dir: str, scene_id_offset: int = 500, top_lidar_only: bool = True):
        self.tfrecord_dir = tfrecord_dir
        self.output_dir = os.path.expanduser(output_dir)
        self.offset = scene_id_offset
        self.top_lidar_only = top_lidar_only
 
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_all(self):
        records = sorted(glob.glob(os.path.join(self.tfrecord_dir, "**", "*.tfrecord"), recursive=True))
        if not records:
            raise FileNotFoundError(f"No .tfrecord files found under {self.tfrecord_dir}")
        print(f"Found {len(records)} TFRecord files.")
        for scene_idx, record_path in enumerate(records):
            self._convert_one(scene_idx + self.offset, record_path)
        print("Done converting all Waymo scenes.")

    def _convert_one(self, scene_id: int, record_path: str):
        seq_dir = os.path.join(self.output_dir, "sequences", f"{scene_id:04d}")
        velo_dir = os.path.join(seq_dir, "velodyne")
        label_dir = os.path.join(seq_dir, "labels")
        os.makedirs(velo_dir,  exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
 
        pose_path = os.path.join(seq_dir, "poses.txt")
        times_path = os.path.join(seq_dir, "times.txt")
        calib_path = os.path.join(seq_dir, "calib.txt")
        weather_path = os.path.join(seq_dir, "weather.txt")
 
        dataset = tf.data.TFRecordDataset(record_path, compression_type="")
        pose_f = open(pose_path, "w")
        times_f = open(times_path, "w")
        weather_f = open(weather_path, "w")
 
        first_pose_inv = None
        time_start = None
        frame_idx = 0
        weather_tag_cache = None

        print(f"  Scene {scene_id:04d} ← {os.path.basename(record_path)}")
 
        for raw in tqdm(dataset, desc=f"    frames", leave=False):
            frame = dp.Frame()
            frame.MergeFromString(raw.numpy())

            if weather_tag_cache is None:
                weather_tag_cache = _weather_tag(frame.context.stats)

            (range_images, camera_projections, seg_labels, range_image_top_pose) = parse_range_image_and_camera_projection(frame)
 
            points, seg_points = self._range_images_to_point_cloud(
                frame, range_images, seg_labels)
 
            if points.shape[0] == 0:
                frame_idx += 1
                continue

            pose_mat = np.array(frame.pose.transform).reshape(4, 4)
            if first_pose_inv is None:
                first_pose_inv = np.linalg.inv(pose_mat)
            rel_pose = first_pose_inv @ pose_mat
            flat = rel_pose[:3].reshape(-1)
            pose_f.write(" ".join(f"{v:.8f}" for v in flat) + "\n")

            ts = frame.timestamp_micros / 1e6
            if time_start is None:
                time_start = ts
            times_f.write(f"{ts - time_start:.6e}\n")

            points.astype(np.float32).tofile(os.path.join(velo_dir, f"{frame_idx:06d}.bin"))

            skitti_labels = np.vectorize(lambda x: WAYMO_TO_SKITTI_LABEL.get(x, 0))(seg_points)
            skitti_labels = np.uint32(skitti_labels)
            skitti_labels.tofile(os.path.join(label_dir, f"{frame_idx:06d}.label"))

            weather_f.write(weather_tag_cache + "\n")
 
            frame_idx += 1
 
        pose_f.close()
        times_f.close()
        weather_f.close()

        with open(calib_path, "w") as f:
            f.write("P0: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write("P1: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write("P2: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write("P3: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
 
        print(f"  → {frame_idx} frames | weather: {weather_tag_cache}")
 
    # ------------------------------------------------------------------
    def _range_images_to_point_cloud(self, frame, range_images, seg_labels):
        """
        Decode range images into (N,4) xyz+intensity and (N,) semantic arrays.
        Uses the Waymo SDK's convert_range_image_to_point_cloud utility.
        Returns points in the *vehicle* (ego) frame.
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
 
        points_list = []
        seg_list = []
 
        for c in calibrations:
            if self.top_lidar_only and c.name != dp.LaserName.TOP:
                continue
 
            if c.name not in range_images:
                continue

            ri = range_images[c.name][0]
            if len(ri.shape.dims) == 0:
                continue

            ri_tensor = tf.reshape(tf.constant(ri.data), ri.shape.dims)

            extrinsic = np.array(c.extrinsic.transform).reshape(4, 4)

            if len(c.beam_inclinations) == 0:
                inclinations = range_image_utils.compute_inclination(tf.constant([c.beam_inclination_min, c.beam_inclination_max]), height=ri_tensor.shape[0])
            else:
                inclinations = tf.constant(c.beam_inclinations)
 
            inclinations = tf.cast(tf.reverse(inclinations, axis=[-1]), tf.float32)

            extrinsic_tf = tf.cast(extrinsic, tf.float32)
            ri_tensor = tf.cast(ri_tensor, tf.float32)
 
            range_image_polar = range_image_utils.compute_range_image_polar(range_image=ri_tensor[..., 0:1], extrinsic=extrinsic_tf, inclination=inclinations)
 
            range_image_cartesian = range_image_utils.compute_range_image_cartesian(range_image_polar=range_image_polar, extrinsic=extrinsic_tf, pixel_pose=None, frame_pose=None)

            mask = ri_tensor[..., 0] > 0

            xyz = tf.boolean_mask(range_image_cartesian[0], mask).numpy()

            intensity = tf.boolean_mask(ri_tensor[..., 1], mask).numpy()

            pts = np.concatenate([xyz, intensity[:, None]], axis=1)
            points_list.append(pts)

            if c.name in seg_labels and len(seg_labels[c.name]) > 0:
                sl = seg_labels[c.name][0]
                sl_tensor = tf.reshape(tf.constant(sl.data), sl.shape.dims)
                sem = tf.boolean_mask(sl_tensor[..., 1], mask).numpy()
            else:
                sem = np.zeros(xyz.shape[0], dtype=np.int32)
 
            seg_list.append(sem)
 
        if points_list:
            return (np.concatenate(points_list, axis=0).astype(np.float32), np.concatenate(seg_list,    axis=0).astype(np.int32))
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

class WaymoDataset(Dataset):
    """
    Reads Waymo data that has been converted to SemanticKITTI layout by
    WaymoConverter.  Folder structure expected:
 
        <root>/sequences/<seq_id:04d>/
            velodyne/   *.bin
            labels/     *.label
            weather.txt (one tag per line, same order as sorted filenames)
 
    Args:
        root            : Dataset root (contains sequences/).
        sequences       : List of integer sequence IDs to load.
        labels          : Label-index → name dict (same as SemanticKitti).
        color_map       : Label-index → BGR colour dict.
        learning_map    : Original label → xentropy index.
        learning_map_inv: xentropy index → original label.
        sensor          : Sensor config dict (fov_up, fov_down, img_prop, …).
        max_points      : Pad / truncate to this many points.
        gt              : Load ground-truth labels.
        transform       : Apply random augmentation (train mode).
        weather_filter  : List of weather strings to include, or None for all.
                          Valid tags: "sunny", "rain", "fog", "night".
    """
    VALID_WEATHER_TAGS = {"sunny", "rain", "fog", "night"}
 
    def __init__(self,
                 root: str,
                 sequences: list,
                 labels: dict,
                 color_map: dict,
                 learning_map: dict,
                 learning_map_inv: dict,
                 sensor: dict,
                 max_points: int = 150000,
                 gt: bool = True,
                 transform: bool = False,
                 weather_filter: list = None):
 
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],  dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform
        self.nclasses = len(learning_map_inv)

        if weather_filter is not None:
            bad = set(weather_filter) - self.VALID_WEATHER_TAGS
            if bad:
                raise ValueError(f"Unknown weather tags: {bad}. Valid: {self.VALID_WEATHER_TAGS}")
        self.weather_filter = set(weather_filter) if weather_filter else None
 
        assert os.path.isdir(self.root), f"Sequences folder not found: {self.root}"
        assert isinstance(self.labels, dict)
        assert isinstance(self.color_map, dict)
        assert isinstance(self.learning_map, dict)
        assert isinstance(self.sequences, list)
 
        self.scan_files = []
        self.label_files = []
 
        for seq_idx in self.sequences:
            seq = None
            for fmt in (f"{int(seq_idx):04d}", f"{int(seq_idx):02d}"):
                candidate = os.path.join(self.root, fmt)
                if os.path.exists(candidate):
                    seq = fmt
                    break
            if seq is None:
                print(f"[WaymoDataset] Warning: sequence {seq_idx} not found, skipping.")
                continue
 
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
 
            raw_scans = sorted([
                os.path.join(dp, f)
                for dp, _, fn in os.walk(os.path.expanduser(scan_path))
                for f in fn if _is_scan(f)
            ])
            raw_labels = sorted([
                os.path.join(dp, f)
                for dp, _, fn in os.walk(os.path.expanduser(label_path))
                for f in fn if _is_label(f)
            ]) if gt else []

            weather_tags = self._load_weather_tags(os.path.join(self.root, seq, "weather.txt"), n_frames=len(raw_scans))

            kept_scans  = []
            kept_labels = []
            kept_n = 0
            for i, scan_f in enumerate(raw_scans):
                tag = weather_tags[i] if i < len(weather_tags) else "sunny"
                if self.weather_filter is None or tag in self.weather_filter:
                    kept_scans.append(scan_f)
                    if gt and raw_labels:
                        kept_labels.append(raw_labels[i])
                    kept_n += 1
 
            print(f"[WaymoDataset] seq {seq}: {kept_n}/{len(raw_scans)} frames kept (filter={self.weather_filter})")
 
            if gt:
                assert len(kept_scans) == len(kept_labels), "Scan/label count mismatch after weather filtering."
 
            self.scan_files.extend(kept_scans)
            self.label_files.extend(kept_labels)
 
        print(f"[WaymoDataset] Total: {len(self.scan_files)} scans from sequences {self.sequences}")

    @staticmethod
    def _load_weather_tags(weather_path: str, n_frames: int) -> list:
        """Read weather.txt; fall back to all-sunny if file is missing."""
        if not os.path.exists(weather_path):
            print(f"[WaymoDataset] weather.txt not found at {weather_path}, "
                  "treating all frames as 'sunny'.")
            return ["sunny"] * n_frames
        with open(weather_path) as f:
            tags = [line.strip() for line in f if line.strip()]
        if len(tags) < n_frames:
            tags += ["sunny"] * (n_frames - len(tags))
        return tags

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        label_file = self.label_files[index] if self.gt else None

        DA = flip_sign = rot = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)
 
        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down,
                                DA=DA,
                                flip_sign=flip_sign,
                                rot=rot,
                                drop_points=drop_points)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down,
                             DA=DA,
                             flip_sign=flip_sign,
                             rot=rot,
                             drop_points=drop_points)
 
        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            scan.sem_label = self.map(scan.sem_label,      self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        unproj_n_points = scan.points.shape[0]
 
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
 
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
 
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
 
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
 
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
 
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

        proj = torch.cat([proj_range.unsqueeze(0).clone(), proj_xyz.clone().permute(2, 0, 1), proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return (proj, proj_mask, proj_labels, unproj_labels,
                path_seq, path_name,
                proj_x, proj_y,
                proj_range, unproj_range,
                proj_xyz, unproj_xyz,
                proj_remission, unproj_remissions,
                unproj_n_points)

    @staticmethod
    def map(label, mapdict):
        """Identical to SemanticKitti.map – reused here to stay self-contained."""
        maxkey = 0
        for key, data in mapdict.items():
            nel = len(data) if isinstance(data, list) else 1
            if key > maxkey:
                maxkey = key
        if isinstance(next(iter(mapdict.values())), list):
            nel = len(next(iter(mapdict.values())))
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print(f"[WaymoDataset] Wrong key {key}")
        return lut[label]
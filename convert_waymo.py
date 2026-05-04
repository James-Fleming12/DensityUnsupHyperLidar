import os
import numpy as np
import dask
import dask.dataframe as dd
from tqdm import tqdm
import tensorflow as tf

from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils

DASK_TMP_DIR = "/mnt/bravo/jmfleming/dask_scratchpad"
os.makedirs(DASK_TMP_DIR, exist_ok=True)

INPUT_PARQUET_DIR = "/mnt/hdd/jmfleming/waymo_raw/training"
OUTPUT_KITTI_DIR = "/mnt/bravo/jmfleming/waymo_skitti"

dask.config.set({"temporary-directory": DASK_TMP_DIR})

WAYMO_TO_SKITTI_LABEL = {
    0:  0,   1:  9,   2:  10,  3:  18,  4:  18,  5:  16,
    6:  15,  7:  13,  8:  0,   9:  17,  10: 0,   11: 0,
    12: 15,  13: 16,  14: 0,   15: 72,  16: 70,  17: 40,
    18: 40,  19: 44,  20: 48,  21: 51,  22: 52,
}

def _weather_tag(weather_str: str, tod_str: str) -> str:
    """
    Map Waymo strings to a canonical condition string.
    Adapted to accept extracted strings directly.
    """
    weather = weather_str.lower() if weather_str else "sunny"
    tod = tod_str.lower() if tod_str else "day"
 
    if "rain" in weather:
        return "rain"
    if "fog" in weather:
        return "fog"
    if "night" in tod or "dawn" in tod or "dusk" in tod:
        return "night"
    return "sunny"

def _read_tag(tag: str, directory: str) -> dd.DataFrame:
    """Lazily read a Waymo parquet component — no compute() called here."""
    return dd.read_parquet(f"{directory}/{tag}/*.parquet")


def _load_stats(parquet_dir: str) -> dict:
    """
    Load the lightweight stats table (weather / time-of-day) all at once.
    This table is small, so a single global compute() is fine.
    """
    stats_dict = {}
    try:
        stats_df = _read_tag("stats", parquet_dir).compute()
        for _, row in stats_df.iterrows():
            seg_name = row.get("key.segment_context_name")
            if not seg_name:
                continue
            w = row.get("weather", "")
            t = row.get("time_of_day", "")
            try:
                stats_comp = v2.StatsComponent.from_dict(row)
                w = getattr(stats_comp, "weather", w)
                t = getattr(stats_comp, "time_of_day", t)
            except Exception:
                pass
            stats_dict[seg_name] = _weather_tag(str(w), str(t))
        print(f"Loaded weather stats for {len(stats_dict)} segments.")
    except Exception as e:
        print(f"Warning: Could not load 'stats', defaulting to sunny. ({e})")
    return stats_dict


def _get_segment_names(parquet_dir: str) -> list:
    """
    Pull only the segment-name column from lidar so we never load point data
    just to enumerate scenes.
    """
    names = (
        _read_tag("lidar", parquet_dir)[
            ["key.segment_context_name", "key.laser_name"]
        ]
        .query("`key.laser_name` == 1")["key.segment_context_name"]
        .unique()
        .compute()
    )
    return list(names)


def _load_segment_df(parquet_dir: str, segment_name: str):
    """
    Load and join all tables for a *single* segment.
    Filtering is applied in Dask before compute() so only the rows we need
    land in memory.
    """
    def filtered(tag, extra_filter=True):
        df = _read_tag(tag, parquet_dir)
        df = df[df["key.segment_context_name"] == segment_name]
        if extra_filter and "key.laser_name" in df.columns:
            df = df[df["key.laser_name"] == 1]
        return df

    lidar_df = filtered("lidar")
    lidar_pose_df = filtered("lidar_pose")
    lidar_calib_df = filtered("lidar_calibration")
    vehicle_pose_df = filtered("vehicle_pose", extra_filter=False)
    lidar_seg_df = filtered("lidar_segmentation")

    df = v2.merge(lidar_df, lidar_pose_df)
    df = v2.merge(df, vehicle_pose_df)
    df = v2.merge(df, lidar_calib_df)
    df = v2.merge(df, lidar_seg_df)

    return df.compute().sort_values("key.frame_timestamp_micros")

def _convert_segment(seg_df, scene_id: int, weather_tag: str, output_dir: str):
    seq_dir = os.path.join(output_dir, "sequences", f"{scene_id:04d}")
    velo_dir = os.path.join(seq_dir, "velodyne")
    label_dir = os.path.join(seq_dir, "labels")
    os.makedirs(velo_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    first_pose_inv = None
    time_start = None
    frame_idx = 0

    with (
        open(os.path.join(seq_dir, "poses.txt"), "w") as pose_f,
        open(os.path.join(seq_dir, "times.txt"), "w") as times_f,
        open(os.path.join(seq_dir, "weather.txt"), "w") as weather_f,
    ):
        for _, row in tqdm(seg_df.iterrows(), total=len(seg_df), leave=False):
            lidar = v2.LiDARComponent.from_dict(row)
            lidar_pose = v2.LiDARPoseComponent.from_dict(row)
            lidar_calib = v2.LiDARCalibrationComponent.from_dict(row)
            vehicle_pose = v2.VehiclePoseComponent.from_dict(row)
            lidar_seg = v2.LiDARSegmentationLabelComponent.from_dict(row)

            pose_mat = np.array(vehicle_pose.transform).reshape(4, 4)
            if first_pose_inv is None:
                first_pose_inv = np.linalg.inv(pose_mat)
            rel_pose = first_pose_inv @ pose_mat
            pose_f.write(" ".join(f"{v:.8f}" for v in rel_pose[:3].reshape(-1)) + "\n")

            ts = lidar.key.frame_timestamp_micros / 1e6
            if time_start is None:
                time_start = ts
            times_f.write(f"{ts - time_start:.6e}\n")

            points_tensor = lidar_utils.convert_range_image_to_point_cloud(
                lidar.range_image_return1,
                lidar_calib,
                lidar_pose.range_image_return1,
                frame_pose=None,
                keep_polar_features=True,
            )

            ri_shape = lidar.range_image_return1.shape.dims
            ri_tensor = tf.reshape(tf.constant(lidar.range_image_return1.values), ri_shape)
            range_image_mask = ri_tensor[..., 0] > 0

            seg_shape = lidar_seg.range_image_return1.shape.dims
            seg_tensor = tf.reshape(tf.constant(lidar_seg.range_image_return1.values), seg_shape)
            seg_points = tf.boolean_mask(seg_tensor[..., 1], range_image_mask).numpy()

            pts_array = points_tensor.numpy()
            kitti_points = np.zeros((pts_array.shape[0], 4), dtype=np.float32)
            kitti_points[:, :3] = pts_array[:, :3]
            kitti_points[:, 3] = pts_array[:, 4]
            kitti_points.tofile(os.path.join(velo_dir, f"{frame_idx:06d}.bin"))

            skitti_labels = np.vectorize(lambda x: WAYMO_TO_SKITTI_LABEL.get(x, 0))(seg_points)
            np.uint32(skitti_labels).tofile(os.path.join(label_dir, f"{frame_idx:06d}.label"))

            weather_f.write(weather_tag + "\n")
            frame_idx += 1

    with open(os.path.join(seq_dir, "calib.txt"), "w") as f:
        f.write("P0: 0 0 0 0 0 0 0 0 0 0 0 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n")


class WaymoParquetConverter:
    def __init__(
        self,
        parquet_dir: str,
        output_dir: str,
        scene_id_offset: int = 500,
        batch_size: int = 10,
    ):
        self.parquet_dir = parquet_dir
        self.output_dir = os.path.expanduser(output_dir)
        self.offset = scene_id_offset
        self.batch_size = batch_size
        self.checkpoint = os.path.join(self.output_dir, "converted.txt")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_done(self) -> set:
        if not os.path.exists(self.checkpoint):
            return set()
        with open(self.checkpoint) as f:
            return {line.strip() for line in f if line.strip()}

    def _mark_done(self, segment_name: str):
        with open(self.checkpoint, "a") as f:
            f.write(segment_name + "\n")

    def convert_all(self):
        print(f"Reading segment names from {self.parquet_dir}...")
        stats_dict = _load_stats(self.parquet_dir)
        all_segments = _get_segment_names(self.parquet_dir)
        done_segments = self._load_done()

        pending = [s for s in all_segments if s not in done_segments]
        print(f"Total: {len(all_segments)}  |  Done: {len(done_segments)}  |  Pending: {len(pending)}")

        for batch_start in range(0, len(pending), self.batch_size):
            batch = pending[batch_start : batch_start + self.batch_size]
            print(f"\n── Batch {batch_start // self.batch_size + 1} (segments {batch_start + 1}–{batch_start + len(batch)} of {len(pending)}) ──")

            for segment_name in batch:
                scene_id = all_segments.index(segment_name) + self.offset
                weather_tag = stats_dict.get(segment_name, "sunny")

                print(f"  → Scene {scene_id:04d}  [{weather_tag}]  {segment_name}")
                try:
                    seg_df = _load_segment_df(self.parquet_dir, segment_name)
                    _convert_segment(seg_df, scene_id, weather_tag, self.output_dir)
                    self._mark_done(segment_name)
                    del seg_df
                except Exception as e:
                    print(f"    ✗ FAILED: {e}")

if __name__ == "__main__":
    converter = WaymoParquetConverter(
        parquet_dir=INPUT_PARQUET_DIR,
        output_dir=OUTPUT_KITTI_DIR,
        scene_id_offset=500,
        batch_size=10,
    )
    converter.convert_all()
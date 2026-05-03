import os
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm
import tensorflow as tf

from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils

# Point this to a folder that contains the /lidar, /lidar_pose, /stats etc. directories
INPUT_PARQUET_DIR = "/path/to/waymo_open_dataset_v_2_0_0/training"
OUTPUT_KITTI_DIR = "./waymo_kitti_format"

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

class WaymoParquetConverter:
    def __init__(self, parquet_dir: str, output_dir: str, scene_id_offset: int = 500):
        self.parquet_dir = parquet_dir
        self.output_dir = os.path.expanduser(output_dir)
        self.offset = scene_id_offset
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_all(self):
        print(f"Reading Parquet metadata from {self.parquet_dir}...")

        def read_tag(tag, directory):
            """Loads a Waymo parquet component using Dask."""
            return dd.read_parquet(f"{directory}/{tag}/*.parquet")

        stats_dict = {}
        try:
            stats_df = read_tag('stats', self.parquet_dir).compute() 
            for _, row in stats_df.iterrows():
                seg_name = row.get('key.segment_context_name')
                if not seg_name:
                    continue

                w = row.get('weather', '')
                t = row.get('time_of_day', '')

                try:
                    stats_comp = v2.StatsComponent.from_dict(row)
                    w = getattr(stats_comp, 'weather', w)
                    t = getattr(stats_comp, 'time_of_day', t)
                except Exception:
                    pass

                stats_dict[seg_name] = _weather_tag(str(w), str(t))
            print(f"Loaded weather statistics for {len(stats_dict)} segments.")
        except Exception as e:
            print(f"Warning: Could not load 'stats' component, defaulting to sunny for all. ({e})")

        lidar_df = read_tag('lidar', self.parquet_dir)
        lidar_pose_df = read_tag('lidar_pose', self.parquet_dir)
        lidar_calib_df = read_tag('lidar_calibration', self.parquet_dir)
        vehicle_pose_df = read_tag('vehicle_pose', self.parquet_dir)
        lidar_seg_df = read_tag('lidar_segmentation', self.parquet_dir)

        lidar_df = lidar_df[lidar_df['key.laser_name'] == 1]
        lidar_pose_df = lidar_pose_df[lidar_pose_df['key.laser_name'] == 1]
        lidar_calib_df = lidar_calib_df[lidar_calib_df['key.laser_name'] == 1]
        lidar_seg_df = lidar_seg_df[lidar_seg_df['key.laser_name'] == 1]

        print("Joining tables...")
        df = v2.merge(lidar_df, lidar_pose_df)
        df = v2.merge(df, vehicle_pose_df)
        df = v2.merge(df, lidar_calib_df)
        df = v2.merge(df, lidar_seg_df)

        segments = df['key.segment_context_name'].unique().compute()
        print(f"Found {len(segments)} segments to convert.")
        
        for i, segment_name in enumerate(segments):
            weather_tag = stats_dict.get(segment_name, "sunny")
            self._convert_segment(df, segment_name, i + self.offset, weather_tag)

    def _convert_segment(self, full_df, segment_name, scene_id, weather_tag):
        seq_dir = os.path.join(self.output_dir, "sequences", f"{scene_id:04d}")
        velo_dir = os.path.join(seq_dir, "velodyne")
        label_dir = os.path.join(seq_dir, "labels")
        os.makedirs(velo_dir,  exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        seg_df = full_df[full_df['key.segment_context_name'] == segment_name].compute()
        seg_df = seg_df.sort_values('key.frame_timestamp_micros')

        pose_f = open(os.path.join(seq_dir, "poses.txt"), "w")
        times_f = open(os.path.join(seq_dir, "times.txt"), "w")
        weather_f = open(os.path.join(seq_dir, "weather.txt"), "w")

        first_pose_inv = None
        time_start = None
        frame_idx = 0

        print(f"  Converting Scene {scene_id:04d} ({segment_name}) - Weather: [{weather_tag}]")

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
            if time_start is None: time_start = ts
            times_f.write(f"{ts - time_start:.6e}\n")

            points_tensor = lidar_utils.convert_range_image_to_point_cloud(
                lidar.range_image_return1,
                lidar_calib,
                lidar_pose.range_image_return1,
                frame_pose=None, 
                keep_polar_features=True 
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

        pose_f.close()
        times_f.close()
        weather_f.close()
        
        with open(os.path.join(seq_dir, "calib.txt"), "w") as f:
            f.write("P0: 0 0 0 0 0 0 0 0 0 0 0 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n")

if __name__ == "__main__":
    converter = WaymoParquetConverter(INPUT_PARQUET_DIR, OUTPUT_KITTI_DIR)
    converter.convert_all()
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesLidarSegDataset(Dataset):
    """
    PyTorch Dataset for nuScenes lidar segmentation.
    """
    
    def __init__(self, 
                 data_path: str,
                 version: str = 'v1.0-mini',
                 split: str = 'train',
                 transform=None):
        """
        Args:
            data_path: Path to nuScenes data directory
            version: Dataset version ('v1.0-mini' for mini)
            split: 'train' or 'val'
            transform: Optional transform to apply to point clouds
        """
        self.data_path = data_path
        self.version = version
        self.split = split
        self.transform = transform

        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.data_path,
            verbose=True
        )

        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load sample tokens for the specified split."""
        scenes = self.nusc.scene
        train_scenes = [scene for scene in scenes if scene['name'].startswith('scene-0061') or 
                       scene['name'].startswith('scene-0553') or 
                       scene['name'].startswith('scene-0655')]

        if self.split == 'train':
            selected_scenes = train_scenes[:2]
        else:
            selected_scenes = train_scenes[2:]
        for scene in selected_scenes:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                self.samples.append(sample_token)

                sample_token = sample['next']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        
        pc_path = os.path.join(self.data_path, lidar_data['filename'])
        points = LidarPointCloud.from_file(pc_path).points.T
        
        lidarseg_path = os.path.join(self.data_path, 
                                    self.nusc.get('lidarseg', lidar_token)['filename'])
        labels = np.fromfile(lidarseg_path, dtype=np.uint8)

        valid_mask = labels > 0
        points = points[valid_mask]
        labels = labels[valid_mask]

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        if self.transform:
            points, labels = self.transform(points, labels)
        
        return {
            'points': points,
            'labels': labels,
            'sample_token': sample_token,
            'scene_token': sample['scene_token']
        }
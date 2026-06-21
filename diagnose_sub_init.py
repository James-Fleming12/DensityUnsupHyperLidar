import yaml
import torch
import numpy as np
import copy
import types
from faster_mean_shift.mean_shift_cosine_gpu import mean_shift_binary

from modules.HDC_utils import DensityModel
from unsup_ugw import get_condition_loaders

def diagnose_sub_init():
    print("Loading configs...")
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    except Exception as e:
        print(f"Error opening config files: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    HDC_SAVE_PATH = "logs/hdc.pth"
    print(f"Loading pretrained model from {HDC_SAVE_PATH}...")
    try:
        model = torch.load(HDC_SAVE_PATH, map_location=device)
        model.eval()
    except Exception as e:
        print(f"Error loading model from {HDC_SAVE_PATH}. Ensure that pretraining has been run. {e}")
        return

    PRE_DATA = copy.deepcopy(DATA)
    PRE_DATA["weather_filter"] = ["sunny"]
    ARCH["train"]["batch_size"] = 6
    
    print("Preparing dataloader...")
    sunny_loaders = get_condition_loaders(
        ARCH, PRE_DATA, PRE_DATA["split"]["train"], 
        batch_size=ARCH["train"]["batch_size"], 
        shuffle=True, 
        conditions=["sunny"]
    )
    
    if "sunny" not in sunny_loaders:
        print("No sunny frames found.")
        return
        
    sunny_loader = sunny_loaders["sunny"]
    
    # Dictionary to store the diagnosis data
    clusters_found_per_class = {}
    
    # We patch _process_single_class to intercept the number of clusters found before FPS reduction
    def patched_process_single_class(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
        if len(class_emb_np) == 0:
            clusters_found_per_class[class_id] = 0
            return []
        
        print(f"  Running mean shift on {len(class_emb_np)} samples...")
        cluster_centers = mean_shift_binary(
            X=class_emb_np,
            bandwidth=bandwidth,
            quantile=self.quantile,
            bandwidth_multiplier=self.mult,
            dedup_scale=self.dedup
        )
        if self.subcluster_type == "bipolar":
            cluster_centers = np.sign(cluster_centers)
        
        num_clusters_found = len(cluster_centers)
        clusters_found_per_class[class_id] = num_clusters_found
        print(f"  [Diagnose] Found {num_clusters_found} clusters for class {class_id}")

        subclusters = []
        if num_clusters_found <= num_sub_per_cluster:
            for center in cluster_centers:
                subclusters.append(torch.tensor(center, device='cpu', dtype=torch.float32))
        else:
            center_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
            fps_indices = self._farthest_point_sample(center_tensor, num_sub_per_cluster)
            for idx in fps_indices.tolist():
                subclusters.append(torch.tensor(cluster_centers[idx], device='cpu', dtype=torch.float32))

        return subclusters

    # Bind the patched method to the model instance
    model._process_single_class = types.MethodType(patched_process_single_class, model)

    print("\nRunning subcluster initialization...")
    # This will now use our patched method and populate clusters_found_per_class
    model.init_subclusters(sunny_loader)
    
    print("\n" + "="*60)
    print("DIAGNOSIS: Mean Shift Clusters Found Per Semantic Class")
    print("="*60)
    
    found_counts = []
    # Using model.num_classes
    for class_id in range(model.num_classes):
        class_name = DATA["labels"].get(class_id, f"Class {class_id}")
        if class_id in clusters_found_per_class:
            count = clusters_found_per_class[class_id]
            print(f"[{class_id:2d}] {class_name:20s}: {count:4d} centers found")
            found_counts.append(count)
        else:
            print(f"[{class_id:2d}] {class_name:20s}:    0 centers found (No data)")
            
    if found_counts:
        mean_clusters = np.mean(found_counts)
        var_clusters = np.var(found_counts)
        std_clusters = np.std(found_counts)
        print("-" * 60)
        print(f"Total active classes: {len(found_counts)} / {model.num_classes}")
        print(f"Mean centers across classes: {mean_clusters:.2f}")
        print(f"Variance across classes:     {var_clusters:.2f}")
        print(f"Std Dev across classes:      {std_clusters:.2f}")
        print(f"Min centers: {np.min(found_counts)} | Max centers: {np.max(found_counts)}")
    print("="*60)

if __name__ == '__main__':
    diagnose_sub_init()

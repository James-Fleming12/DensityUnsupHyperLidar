import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from modules.aug_model import AugModel
from dataset.kitti.parser import Parser
from collections import defaultdict

def test_occupancy(dataset_root, pretrained_path, yaml_labels, yaml_arch, device, output_dir):
    DATA = yaml.safe_load(open(yaml_labels, 'r'))
    ARCH = yaml.safe_load(open(yaml_arch, 'r'))
    
    parser_obj = Parser(root=dataset_root,
                        train_sequences=DATA["split"]["valid"],
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA.get("color_map", {}),
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=1,
                        workers=8,
                        gt=True,
                        shuffle_train=False)
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False, num_workers=8)
    
    # Load model
    modeldir = os.path.dirname(pretrained_path)
    model = AugModel(ARCH, modeldir, 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
    model = model.to(device)
    model.eval()
    
    num_subclusters = ARCH["model"]["num_subclusters"]
    
    # Track occupancy per class.
    # occupancy[class_id] will be an array of size `num_subclusters`
    occupancy = {c: np.zeros(num_subclusters, dtype=np.int64) for c in range(17)}
    
    print(f"\n--- Running D-Occupancy Test on clean validation data ---")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            proj_in = batch_data[0].to(device)
            oracle_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] == 0:
                continue
                
            enc_base, _, _ = model.encode(proj_in)
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            if not torch.any(valid_enc_mask):
                continue
                
            raw_base = F.normalize(enc_base[valid_enc_mask])
            active_oracle = oracle_labels.reshape(-1)[valid_enc_mask]
            
            prototypes = F.normalize(model.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)
            
            S_base = raw_base @ prototypes.T
            preds = S_base.argmax(dim=1)
            
            # We care about the points genuinely belonging to the class (oracle) 
            # or predicted as the class. Let's look at correctly predicted points 
            # to see which subclusters are actually active for true instances.
            valid_mask = (active_oracle > 0) & (active_oracle < 17) & (preds == active_oracle)
            
            if not torch.any(valid_mask):
                continue
                
            filtered_preds = preds[valid_mask]
            filtered_raw_base = raw_base[valid_mask]
            
            for c_id in torch.unique(filtered_preds):
                c_id_item = c_id.item()
                c_mask = (filtered_preds == c_id)
                c_encs = filtered_raw_base[c_mask]
                
                # Get the subclusters for this class
                sub_mask = model.subcluster_to_class == c_id_item
                c_subs = F.normalize(model.subclusters[sub_mask].float(), dim=1)
                
                # Compute similarity of these points to the K subclusters
                c_encs = c_encs.float() # ensure fp32 for matmul
                S_subs = c_encs @ c_subs.T
                
                # Find winning subcluster (argmax)
                winning_subs = S_subs.argmax(dim=1).cpu().numpy()
                
                unique, counts = np.unique(winning_subs, return_counts=True)
                for u, count in zip(unique, counts):
                    occupancy[c_id_item][u] += count
            
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"Processed {batch_idx} frames...")
                
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== OCCUPANCY RESULTS ===")
    for c_id in range(1, 17):
        counts = occupancy[c_id]
        total = counts.sum()
        if total == 0:
            continue
            
        proportions = counts / total
        max_prop = proportions.max()
        max_idx = proportions.argmax()
        
        # Sort for printing
        sorted_props = np.sort(proportions)[::-1]
        active_modes = (proportions > 0.01).sum() # >1% of points
        
        print(f"Class {c_id:2d}: {total:8d} points | Max Mode = {max_prop:.1%} (Idx {max_idx}) | Modes >1%: {active_modes}/{num_subclusters}")
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.bar(range(num_subclusters), proportions)
        plt.title(f'Subcluster Occupancy - Class {c_id} (Total: {total})')
        plt.xlabel('Subcluster Index')
        plt.ylabel('Fraction of Points')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        out_path = os.path.join(output_dir, f'occupancy_class_{c_id}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"\nSaved histograms to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth')
    parser.add_argument('--output_dir', type=str, default='logs/occupancy_tests')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_labels = 'config/labels/semantic-kitti-all.yaml'
    yaml_arch = 'config/arch/senet-2048p.yml'
    
    test_occupancy(args.kitti_dir, args.pretrained_path, yaml_labels, yaml_arch, device, args.output_dir)

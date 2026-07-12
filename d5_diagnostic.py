import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset.kitti.parser import Parser
from modules.aug_model import AugModel
from unsup_kitti_c import load_hdc_model, NUM_CLASSES
import yaml

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    with open('config/kitti-c.yaml', 'r') as f:
        ARCH = yaml.safe_load(f)
    with open('config/semantic-kitti.yaml', 'r') as f:
        DATA = yaml.safe_load(f)
        
    data_dir = '/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/wet_ground/heavy'
    if not os.path.exists(data_dir):
        print(f"Dataset path not found: {data_dir}. Diagnostic must be run on the cluster.")
        return
        
    # Initialize parser
    parser = Parser(root=data_dir,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=[8],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=1,
                    workers=4,
                    gt=True,
                    shuffle_train=False)
                    
    # Load model
    pretrained_path = 'logs/kitti_pretrain/hdc_sub.pth'
    hdc_model = load_hdc_model(pretrained_path, num_classes=NUM_CLASSES)
    model = AugModel(hdc_model).to(device)
    model.eval()

    dataloader = parser.get_valid_set()
    
    # Get first batch
    for batch_data in dataloader:
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[1].to(device)
        proj_xyz = batch_data[4].to(device) if len(batch_data) > 4 else None
        break

    print("Running D5 Diagnostic on first batch of wet_ground-3...")
    
    with torch.no_grad():
        x = proj_in
        enc, _, _ = model.base_model.encode(x)
        original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
        valid_enc_mask = torch.any(original_x != 0, dim=1)
        
        raw_base = F.normalize(enc[valid_enc_mask])
        
        # Bundling
        bundled_target = raw_base.clone()
        x_yaw = torch.roll(x, shifts=14, dims=3)
        enc_yaw, _, _ = model.base_model.encode(x_yaw)
        bundled_target.add_(F.normalize(enc_yaw[valid_enc_mask]))
        
        x_scale = x * 0.95
        enc_scale, _, _ = model.base_model.encode(x_scale)
        bundled_target.add_(F.normalize(enc_scale[valid_enc_mask]))
        
        bundled = F.normalize(bundled_target)
        
        # Get Predictions
        prototypes = F.normalize(model.base_model.classify.weight.float())
        preds = (bundled @ prototypes.T).argmax(dim=1)
        
        # Compute Similarities to Nearest Subcluster
        raw_sims = []
        bundled_sims = []
        
        unique_classes = torch.unique(preds)
        for c in unique_classes:
            c_id = c.item()
            mask = (preds == c_id)
            if not torch.any(mask): continue
            
            raw_c = raw_base[mask]
            bund_c = bundled[mask]
            
            # Subcluster similarities
            raw_sub, _ = model.base_model.get_max_subcluster_similarity(raw_c, c_id, distance_sensitivity=1.0)
            bund_sub, _ = model.base_model.get_max_subcluster_similarity(bund_c, c_id, distance_sensitivity=1.0)
            
            raw_sims.append(raw_sub)
            bundled_sims.append(bund_sub)
            
        raw_sims = torch.cat(raw_sims).cpu().numpy()
        bundled_sims = torch.cat(bundled_sims).cpu().numpy()
        
        print(f"Mean Similarity (Single View vs Subcluster): {np.mean(raw_sims):.4f}")
        print(f"Mean Similarity (Bundled vs Subcluster): {np.mean(bundled_sims):.4f}")
        print(f"Median Similarity (Single View vs Subcluster): {np.median(raw_sims):.4f}")
        print(f"Median Similarity (Bundled vs Subcluster): {np.median(bundled_sims):.4f}")
        
        # Plot Histograms
        plt.figure(figsize=(10, 6))
        plt.hist(raw_sims, bins=50, alpha=0.5, label=f'Single View (Mean: {np.mean(raw_sims):.3f})')
        plt.hist(bundled_sims, bins=50, alpha=0.5, label=f'Bundled Views (Mean: {np.mean(bundled_sims):.3f})')
        plt.title('D5 Diagnostic: Subcluster Similarity Collapses with View Bundling?')
        plt.xlabel('Cosine Similarity to Nearest Subcluster')
        plt.ylabel('Point Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('logs/kitti_c_test/d5_diagnostic_hist.png')
        print("Saved diagnostic plot to logs/kitti_c_test/d5_diagnostic_hist.png")

if __name__ == '__main__':
    main()

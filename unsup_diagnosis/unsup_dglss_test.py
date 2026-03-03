import torch
import yaml

from dataset.kitti.parser import Parser
from faster_mean_shift.mean_shift_cosine_gpu import estimate_bandwidth_binary, mean_shift_binary
from modules.HDC_utils import DensityModel
from modules.trainer import Trainer
from modules.ioueval import iouEval

import numpy as np
import torch.nn.functional as F

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 20
FEATURE_EXTRACTOR_EPOCHS = 400

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def diag_feature_variance(model, dataloader):
    """Diagnoses if the CNN is producing diverse features or collapsing."""
    device = model.device
    all_means = []
    all_vars = []

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            _, _, feat_map = model.net(proj_in.to(device)) 

            feat_flat = feat_map.permute(0, 2, 3, 1).reshape(-1, feat_map.shape[1])
            feat_norm = F.normalize(feat_flat, p=2, dim=1)

            variance = torch.var(feat_norm, dim=0).mean()
            all_vars.append(variance.item())

    print("\n[CNN Feature Manifold Health]")
    avg_var = sum(all_vars) / len(all_vars)
    print(f"Average Feature Variance: {avg_var:.6f}")
    if avg_var < 1e-4:
        print("!!! WARNING: Manifold Collapse detected. Your CNN is outputting nearly identical vectors for all points.")

def diag_class_orthogonality(model):
    """Checks the cosine similarity matrix between class hypervectors."""
    class_hvs = F.normalize(model.classify.weight.data, p=2, dim=1)

    sim_matrix = class_hvs @ class_hvs.T
    
    print("\n[Class Hypervector Orthogonality]")
    mask = ~torch.eye(model.num_classes, device=sim_matrix.device).bool()
    avg_inter_sim = sim_matrix[mask].mean().item()
    max_inter_sim = sim_matrix[mask].max().item()

    print(f"Average Inter-class Similarity: {avg_inter_sim:.3f}")
    print(f"Maximum Inter-class Similarity: {max_inter_sim:.3f}")
    
    if avg_inter_sim > 0.4:
        print("!!! WARNING: High Class Correlation. Hypervectors are not orthogonal enough.")

def diag_sparsity_invariance_gap(model, dataloader):
    """Measures the actual angular shift caused by beam dropping."""
    device = model.device
    angular_shifts = []

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            
            _, _, z8_dense = model.net(proj_in)
            z8_dense = F.normalize(z8_dense, p=2, dim=1)

            in_sparse = proj_in.clone()
            in_sparse[:, :, ::2, :] = 0 
            _, _, z8_sparse = model.net(in_sparse)
            z8_sparse = F.normalize(z8_sparse, p=2, dim=1)

            cos_dist = 1.0 - (z8_dense * z8_sparse).sum(dim=1)
            angular_shifts.append(cos_dist.mean().item())

    print("\n[DGLSS Sparsity Invariance Gap]")
    avg_gap = sum(angular_shifts) / len(angular_shifts)
    print(f"Mean Angular Shift (Dense vs 50% Sparse): {avg_gap:.4f}")
    if avg_gap > 0.2:
        print("!!! WARNING: SIFC is failing. Features are not density-invariant.")

def diag_subcluster_distinction(model):
    """Diagnoses if subclusters within the same class are too similar."""
    print("\n[Subcluster Distinction per Class]")
    for c in range(model.num_classes):
        mask = model.subcluster_to_class == c
        subs = model.subclusters[mask]
        
        if subs.shape[0] < 2:
            continue
            
        subs_norm = F.normalize(subs, p=2, dim=1)
        sim_matrix = subs_norm @ subs_norm.T

        m = ~torch.eye(subs.shape[0], device=sim_matrix.device).bool()
        avg_internal_sim = sim_matrix[m].mean().item()
        
        print(f"Class {c:2d} | Internal Subcluster Similarity: {avg_internal_sim:.3f}")
        if avg_internal_sim > 0.8:
            print(f"  -> WARNING: Subclusters for class {c} are nearly identical.")

def diag_rp_orthogonality(model):
    """Checks if the RP matrix is causing dimensional interference."""
    with torch.no_grad():
        W = model.projection.weight.data
        gram = W @ W.T 

        diag = torch.diag(gram)
        norm_gram = gram / diag.unsqueeze(1)

        identity = torch.eye(W.shape[0], device=W.device)
        interference = torch.abs(norm_gram - identity).mean().item()
        
    print("\n[RP Matrix Topology]")
    print(f"Mean Dimensional Interference: {interference:.4f}")
    if interference > 0.15:
        print("!!! WARNING: RP matrix is not orthogonal. Dimensions are 'bleeding' into each other.")

def diag_hd_bit_entropy(model, dataloader):
    """Measures how much information is surviving the RP + Quantization process."""
    device = model.device
    bit_counts = torch.zeros(model.hd_dim, device=device)
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            hvs, _, _ = model.encode(proj_in.to(device))

            bit_counts += (hvs > 0).float().sum(dim=0)
            total_samples += hvs.shape[0]

    p_high = bit_counts / total_samples
    entropy = - (p_high * torch.log2(p_high + 1e-8) + (1 - p_high) * torch.log2(1 - p_high + 1e-8))
    mean_entropy = entropy.mean().item()

    print("\n[HD Information Bottleneck]")
    print(f"Mean Bit Entropy: {mean_entropy:.4f} (Ideal: 1.000)")
    if mean_entropy < 0.85:
        print("!!! WARNING: Information Loss. The RP/Quantization is producing 'frozen' bits.")

def diag_mapping_distortion(model, dataloader):
    """Measures if RP is preserving the distances learned by the DGLSS trainer."""
    device = model.device
    distortions = []

    model.eval()
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            _, _, feat_map = model.net(proj_in.to(device))
            x = feat_map.permute(0, 2, 3, 1).reshape(-1, 128)
            x_norm = F.normalize(x, p=2, dim=1)
            
            hvs, _, _ = model.encode(proj_in.to(device))
            hvs_norm = F.normalize(hvs.float(), p=2, dim=1)

            idx1 = torch.randperm(x.shape[0])[:100]
            idx2 = torch.randperm(x.shape[0])[:100]
            
            cnn_sim = (x_norm[idx1] * x_norm[idx2]).sum(dim=1)
            hd_sim = (hvs_norm[idx1] * hvs_norm[idx2]).sum(dim=1)

            distortion = torch.abs(cnn_sim - hd_sim).mean().item()
            distortions.append(distortion)

    print("\n[RP Mapping Distortion]")
    avg_dist = sum(distortions) / len(distortions)
    print(f"Mean Distance Distortion: {avg_dist:.4f}")
    if avg_dist > 0.3:
        print("!!! WARNING: RP is smearing features. Distances in HD space do not match CNN logic.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_mini.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    parser = Parser(root=DATA_DIR,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    
    dataloader = parser.get_train_set()

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    # diag_feature_variance(model, dataloader)
    # diag_class_orthogonality(model)
    # diag_sparsity_invariance_gap(model, dataloader)
    # diag_subcluster_distinction(model)

    diag_rp_orthogonality(model)
    diag_hd_bit_entropy(model, dataloader)
    diag_mapping_distortion(model, dataloader)

if __name__=="__main__":
    main()
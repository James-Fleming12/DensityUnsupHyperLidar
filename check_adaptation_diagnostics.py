import torch
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.aug_model import AugModel
from dataset.kitti.parser import Parser
from torch.utils.data import DataLoader
import torch.nn.functional as F

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA_NUSC = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    KITTI_LABELS = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["labels"]
    inv_map = DATA_NUSC["learning_map_inv"]
    
    pretrained_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = AugModel(ARCH, os.path.dirname(pretrained_path), 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    model.eval()

    nusc_sensor = ARCH["dataset"]["sensor"].copy()
    nusc_sensor["fov_up"] = 10.0
    nusc_sensor["fov_down"] = -30.0
    nusc_sensor["img_prop"] = nusc_sensor["img_prop"].copy()
    nusc_sensor["img_prop"]["height"] = 32
    nusc_sensor["img_prop"]["width"] = 1024

    print("Initializing NuScenes Dataset...")
    parser_obj = Parser(root="/mnt/alpha/jmfleming/nuscenes_kitti",
                        train_sequences=[854], valid_sequences=[854], test_sequences=None,
                        labels=DATA_NUSC["labels"], color_map=DATA_NUSC.get("color_map", {}),
                        learning_map=DATA_NUSC["learning_map"], learning_map_inv=DATA_NUSC["learning_map_inv"],
                        sensor=nusc_sensor, max_points=ARCH["dataset"]["max_points"],
                        batch_size=1, workers=1, gt=True, shuffle_train=False)
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False)

    print("\nRunning Gating Diagnostics on NuScenes (10 Frames)...")
    
    total_gated = 0
    total_correct_gated = 0
    
    class_gated_counts = torch.zeros(17, device=device)
    
    correct_sims = []
    incorrect_sims = []
    
    correct_proto_sims = []
    incorrect_proto_sims = []

    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= 10:
            break
            
        proj_in = batch_data[0].to(device)
        proj_xyz = batch_data[10].to(device) if len(batch_data) > 10 else None
        proj_labels = batch_data[2].to(device).view(-1)
        
        with torch.no_grad():
            # Standard HDC Forward
            enc_base, _, _ = model.encode(proj_in)
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            raw_base = F.normalize(enc_base[valid_enc_mask])
            
            prototypes = F.normalize(model.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)
            preds = (raw_base @ prototypes.T).argmax(dim=1)
            
            # Subcluster Similarity Calculation
            sub_norm = F.normalize(model.subclusters.to(raw_base.dtype), dim=1)
            cosine_sim = raw_base @ sub_norm.T
            base_similarity = (cosine_sim + 1) / 2
            
            valid_mask = (model.subcluster_to_class.unsqueeze(0) == preds.unsqueeze(1))
            masked_similarity = torch.where(valid_mask, base_similarity, torch.tensor(0.0, device=device))
            sub_sims, _ = torch.max(masked_similarity, dim=1)
            
            # PROTOTYPE SIMILARITY CALCULATION
            proto_sims = (raw_base @ prototypes.T).gather(1, preds.unsqueeze(1)).squeeze(1)
            
            # Simulated Gating Logic
            gate_sims = sub_sims * 2.0 - 1.0
            
            # Oracle Check
            valid_indices = torch.nonzero(valid_enc_mask).squeeze(1)
            active_labels = proj_labels[valid_indices]
            ignore_mask = (active_labels == 0)
            
            # Diagnostic Bypass: Let's see what the actual similarities are for semantic classes!
            semantic_mask = (preds != 0) & ~ignore_mask
            
            if semantic_mask.sum() > 0:
                is_correct = (preds[semantic_mask] == active_labels[semantic_mask])
                
                # Log similarities for correct/incorrect regardless of threshold
                gated_sub_sims = gate_sims[semantic_mask]
                gated_proto_sims = proto_sims[semantic_mask]
                
                correct_sims.append(gated_sub_sims[is_correct].mean().item() if is_correct.sum() > 0 else 0)
                incorrect_sims.append(gated_sub_sims[~is_correct].mean().item() if (~is_correct).sum() > 0 else 0)
                
                correct_proto_sims.append(gated_proto_sims[is_correct].mean().item() if is_correct.sum() > 0 else 0)
                incorrect_proto_sims.append(gated_proto_sims[~is_correct].mean().item() if (~is_correct).sum() > 0 else 0)
                
                # Also we'll just force the update mask to everything so we can see the class distribution
                update_mask = semantic_mask
                
                total_gated += update_mask.sum().item()
                total_correct_gated += is_correct.sum().item()
                
                # Class imbalance check
                gated_preds = preds[update_mask]
                class_gated_counts += torch.bincount(gated_preds, minlength=17)

    print("\n" + "="*50)
    print("DIAGNOSTIC RESULTS: WHY IS ADAPTATION FAILING?")
    print("="*50)
    
    # 1. Purity
    purity = (total_correct_gated / total_gated * 100) if total_gated > 0 else 0
    print(f"\n1. PSEUDO-LABEL PURITY: {purity:.2f}%")
    print("   (Percentage of points the model 'trains' on that are actually correct.)")
    if purity < 80.0:
        print("   >>> FAIL: The gating mechanism is letting in too much garbage. The threshold (0.35) is too low.")
        
    # 2. Class Imbalance
    print(f"\n2. CLASS IMBALANCE IN UPDATES:")
    top_classes = torch.argsort(class_gated_counts, descending=True)[:3]
    for c_idx in top_classes:
        c_val = class_gated_counts[c_idx].item()
        if c_val > 0:
            c_name = KITTI_LABELS.get(inv_map.get(c_idx.item(), 0), f"Class {c_idx}")
            pct = c_val / total_gated * 100
            print(f"   - {c_name}: {pct:.1f}% of all gradients")
    if class_gated_counts[top_classes[0]] / total_gated > 0.8:
        print("   >>> FAIL: The model is exclusively training on one class (e.g., roads).")
        print("   >>> This causes 'Catastrophic Forgetting' where it forgets cars and pedestrians.")
        print("   >>> Fix: We need Class-Balanced Weighting or separate per-class thresholds.")
        
    # 3. Subcluster Health
    mean_corr = sum(correct_sims)/len([x for x in correct_sims if x != 0]) if any(x != 0 for x in correct_sims) else 0
    mean_incorr = sum(incorrect_sims)/len([x for x in incorrect_sims if x != 0]) if any(x != 0 for x in incorrect_sims) else 0
    
    mean_corr_proto = sum(correct_proto_sims)/len([x for x in correct_proto_sims if x != 0]) if any(x != 0 for x in correct_proto_sims) else 0
    mean_incorr_proto = sum(incorrect_proto_sims)/len([x for x in incorrect_proto_sims if x != 0]) if any(x != 0 for x in incorrect_proto_sims) else 0
    
    print(f"\n3. SUBCLUSTER vs PROTOTYPE HEALTH:")
    print(f"   [Subclusters (KITTI space)]")
    print(f"   - Avg Similarity of CORRECT points:   {mean_corr:.3f}")
    print(f"   - Avg Similarity of INCORRECT points: {mean_incorr:.3f}")
    
    print(f"   [Prototypes (Linear Layer)]")
    print(f"   - Avg Similarity of CORRECT points:   {mean_corr_proto:.3f}")
    print(f"   - Avg Similarity of INCORRECT points: {mean_incorr_proto:.3f}")

if __name__ == "__main__":
    main()

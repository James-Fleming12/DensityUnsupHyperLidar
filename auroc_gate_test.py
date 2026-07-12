import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from modules.HDC_utils import HDCSegmentationModel
from datasets.semantic_kitti.parser import Parser

def test_auroc_on_chunk(corruption_root, pretrained_path, yaml_labels, yaml_arch, device, corruption_name):
    # Load dataset
    DATA = yaml.safe_load(open(yaml_labels, 'r'))
    ARCH = yaml.safe_load(open(yaml_arch, 'r'))
    
    parser_obj = Parser(root=corruption_root,
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
    model = HDCSegmentationModel(17, hd_dim=ARCH["model"]["hd_dim"], 
                                 num_subclusters=ARCH["model"]["num_subclusters"],
                                 subcluster_type=ARCH["model"]["subcluster_type"])
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
    model = model.to(device)
    model.eval()
    
    all_is_correct = []
    all_proto_sims = []
    all_sub_sims = []
    
    print(f"\n--- Running AUROC Gate-Quality Test on {corruption_name} ---")
    
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
            
            # Predict
            prototypes = F.normalize(model.classify.weight)
            S_base = raw_base @ prototypes.T
            preds = S_base.argmax(dim=1)
            
            # Filter valid (labels mapped to 0-16, ignore unlabeled/unknown)
            valid_labels_mask = (active_oracle > 0) & (active_oracle < 17) & (preds > 0)
            
            if not torch.any(valid_labels_mask):
                continue
                
            filtered_preds = preds[valid_labels_mask]
            filtered_oracle = active_oracle[valid_labels_mask]
            filtered_raw_base = raw_base[valid_labels_mask]
            
            is_correct = (filtered_preds == filtered_oracle).cpu().numpy()
            all_is_correct.extend(is_correct)
            
            # 1. Prototype Similarity for predicted class
            selected_proto = prototypes[filtered_preds]
            proto_sims = torch.sum(filtered_raw_base * selected_proto, dim=1).cpu().numpy()
            all_proto_sims.extend(proto_sims)
            
            # 2. Subcluster Similarity for predicted class
            unique_preds = torch.unique(filtered_preds)
            sub_sims = torch.zeros(filtered_raw_base.shape[0], device=device)
            for c_id in unique_preds:
                c_id_item = c_id.item()
                c_mask = (filtered_preds == c_id)
                c_encs = filtered_raw_base[c_mask]
                c_sub_sims, _ = model.get_max_subcluster_similarity(c_encs, c_id_item, distance_sensitivity=1.0)
                sub_sims[c_mask] = c_sub_sims
            all_sub_sims.extend(sub_sims.cpu().numpy())
            
            if batch_idx > 0 and batch_idx % 50 == 0:
                print(f"Processed {batch_idx} frames...")
                
    if len(all_is_correct) == 0:
        print("No valid points found to evaluate.")
        return
        
    try:
        proto_auroc = roc_auc_score(all_is_correct, all_proto_sims)
        sub_auroc = roc_auc_score(all_is_correct, all_sub_sims)
        
        print("\n=== AUROC RESULTS ===")
        print(f"Total Points Evaluated: {len(all_is_correct)}")
        print(f"Base Accuracy of Pseudo-Labels: {sum(all_is_correct) / len(all_is_correct):.2%}")
        print(f"AUROC using Prototype Similarity:   {proto_auroc:.4f}")
        print(f"AUROC using Subcluster Similarity: {sub_auroc:.4f}")
        print("=====================\n")
    except ValueError as e:
        print(f"Could not calculate AUROC: {e}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_labels = 'configs/semantic-kitti-all.yaml'
    yaml_arch = 'configs/semantic-kitti.yaml'
    pretrained_path = 'logs/kitti_pretrain/hdc_sub.pth'
    
    # Test on wet_ground severe
    corruption_root = '/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/wet_ground/severe'
    test_auroc_on_chunk(corruption_root, pretrained_path, yaml_labels, yaml_arch, device, 'wet_ground')

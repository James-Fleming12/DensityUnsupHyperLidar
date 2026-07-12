import os
import types
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from modules.HDC_utils import HDCSegmentationModel
from unsup_kitti_c import evaluate_and_adapt, setup_logger, NUM_CLASSES
from datasets.semantic_kitti.parser import Parser

def inference_update_ablation(self, x, beta=0.2, distance_sensitivity=3.0, learning_rate=0.01, thresholds=[0.35, 0.65], proj_xyz=None, level=0, **kwargs):
    self.eval()
    with torch.no_grad():
        enc, _, _ = self.encode(x)
        num_total_samples = enc.shape[0]

        original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
        valid_enc_mask = torch.any(original_x != 0, dim=1)
        
        if not torch.any(valid_enc_mask):
            return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        
        active_enc = enc[valid_enc_mask]
        enc_norm = F.normalize(active_enc)
        
        if enc_norm.dtype != self.classify.weight.dtype:
            enc_norm = enc_norm.to(self.classify.weight.dtype)

        num_active = active_enc.shape[0]
        all_predictions = []
        all_update_masks = []

        # Predict
        for i in range(0, num_active, num_active):
            chunk_enc_norm = enc_norm[i:i+num_active]
            
            sim_to_protos = torch.matmul(chunk_enc_norm, self.classify.weight.T)
            chunk_preds = torch.argmax(sim_to_protos, dim=1)
            all_predictions.append(chunk_preds)
            
            selected_proto = F.normalize(self.classify.weight[chunk_preds])
            sims = torch.sum(chunk_enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            all_update_masks.append(distances > beta)

        predictions = torch.cat(all_predictions)
        update_mask = torch.cat(all_update_masks)
        
        full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        full_predictions[valid_enc_mask] = predictions

        if not torch.any(update_mask):
            return full_predictions

        valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
        unique_classes = torch.unique(predictions[valid_indices_in_active])

        for class_id in unique_classes:
            c_id = class_id.item()
            class_mask = (predictions == c_id) & update_mask
            
            sample_encs = enc_norm[class_mask]

            if self.subcluster_type == 'bipolar':
                target_encs = torch.sign(active_enc[class_mask])
                sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
            else:
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

            if level >= 1:
                valid_mask = (sub_sims > thresholds[0]) & (sub_sims < thresholds[1])
            else:
                valid_mask = sub_sims > 0.45 # density hard gate

            if not torch.any(valid_mask):
                continue

            sample_encs = sample_encs[valid_mask]
            sub_sims = sub_sims[valid_mask]

            if level >= 3 and proj_xyz is not None:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask][class_mask][valid_mask]
                depth = torch.norm(flat_xyz, dim=1)
                # The inverted depth_scale from soft_consensus
                depth_scale = torch.clamp(15.0 / (depth + 1e-3), min=1.0/distance_sensitivity, max=1.0)
            else:
                depth_scale = torch.ones_like(sub_sims)

            if level >= 2:
                conf_weights = torch.clamp((sub_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * depth_scale
            else:
                final_weights = sub_sims * depth_scale

            current_weight = self.classify.weight[c_id]
            
            if level >= 4:
                # soft-consensus's volume_scale formulation (pull_vector is mean)
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                updated_weight = current_weight + learning_rate * pull_vector
            else:
                # density's normalized formulation
                weights = final_weights / (final_weights.sum() + 1e-6)
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * final_weights.mean().item()
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]

            if level >= 5:
                if not hasattr(self, 'source_prototypes'):
                    self.source_prototypes = self.classify.weight.detach().clone()
                anchor_pull = self.source_prototypes[c_id] - current_weight
                anchor_strength = 0.1 * learning_rate
                updated_weight = updated_weight + anchor_strength * anchor_pull

            updated_weight_norm = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
            self.classify.weight[c_id] = updated_weight_norm

        return full_predictions


def main():
    logger = setup_logger('logs/kitti_c_test/ablation.log')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    DATA = yaml.safe_load(open('configs/semantic-kitti-all.yaml', 'r'))
    ARCH = yaml.safe_load(open('configs/semantic-kitti.yaml', 'r'))
    
    corruption_root = '/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/wet_ground/severe'
    
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
    
    print("Running Convergence Ablation on wet_ground (sev 3)...")
    
    for level in range(6):
        print(f"\n--- Running Level {level} ---")
        
        # Fresh model
        model = HDCSegmentationModel(17, hd_dim=ARCH["model"]["hd_dim"], 
                                     num_subclusters=ARCH["model"]["num_subclusters"],
                                     subcluster_type=ARCH["model"]["subcluster_type"])
        checkpoint = torch.load('logs/kitti_pretrain/hdc_sub.pth', map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.inference_update_ablation = types.MethodType(inference_update_ablation, model)
        
        def custom_update_fn(m, x, proj_xyz=None, **kwargs):
            return m.inference_update_ablation(x, proj_xyz=proj_xyz, level=level)
        
        # Pass 1: True Initial
        init_metrics = evaluate_and_adapt(model, dataloader, device, eval_only=True)
        initial_miou = init_metrics["mIoU"][-1]
        
        # Pass 2: Adapt
        adapt_metrics = evaluate_and_adapt(model, dataloader, device, eval_only=False, update_method='custom', custom_update_fn=custom_update_fn)
        
        # Pass 3: True Final
        final_metrics = evaluate_and_adapt(model, dataloader, device, eval_only=True)
        final_miou = final_metrics["mIoU"][-1]
        
        delta = final_miou - initial_miou
        print(f"Level {level}: Initial mIoU={initial_miou:.4f} -> Final={final_miou:.4f} (Delta: {delta:+.4f})")

if __name__ == "__main__":
    main()

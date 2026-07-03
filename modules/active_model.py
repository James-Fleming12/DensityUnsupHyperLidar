import torch
import torch.nn.functional as F
from modules.HDC_utils import DensityModel

class ActiveModel(DensityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize storage for Oracle Subclusters
        self.register_buffer('oracle_subclusters', torch.empty((0, self.hd_dim), dtype=torch.float32, device=self.device))
        self.register_buffer('oracle_subcluster_labels', torch.empty((0,), dtype=torch.long, device=self.device))

    def inference_update_ooa(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Density-Filtered Outliers (Outlier Oracle Anchor) Active Domain Adaptation"""
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
                
            num_active = enc_norm.shape[0]
            
            # Predictions logic
            # Use oracle subclusters if they exist
            chunk_logits = self.classify(enc_norm)
            preds = torch.argmax(chunk_logits, dim=1)
            
            # Removed hard override prediction logic to maintain smooth Voronoi boundaries
                
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            # Active domain adaptation logic
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                
                # 1. Density filter
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    # number of neighbors within 0.5m radius
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                # 2. Highest distance from known subclusters
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    
                    # Also consider distance to existing oracle subclusters
                    if self.oracle_subclusters.shape[0] > 0:
                        sims_oracle_sub = enc_norm @ F.normalize(self.oracle_subclusters).T.to(enc_norm.dtype)
                        max_sims_oracle_sub, _ = sims_oracle_sub.max(dim=1)
                        max_sims_sub = torch.maximum(max_sims_sub, max_sims_oracle_sub)
                    
                    # Ignore points with invalid density by setting their similarity very high
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    # Point with the highest distance (lowest max similarity)
                    outlier_idx = max_sims_sub.argmin()
                    
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        # Register new oracle subcluster
                        new_subcluster = enc_norm[outlier_idx].unsqueeze(0)
                        self.oracle_subclusters = torch.cat([self.oracle_subclusters, new_subcluster], dim=0)
                        self.oracle_subcluster_labels = torch.cat([self.oracle_subcluster_labels, torch.tensor([oracle_label], device=self.device)])
                        
                        # Soft HDC Integration: bundle it directly into the target class's prototype with standard mass weighting
                        self.classify.weight[oracle_label] = F.normalize(
                            self.classify.weight[oracle_label] + new_subcluster.squeeze(0), dim=0
                        )
                        
            # Standard pull update
            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

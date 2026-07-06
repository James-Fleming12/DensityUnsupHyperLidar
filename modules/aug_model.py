import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from modules.HDC_utils import DensityModel
from modules.Basic_HD import DensityTrainer
import torch.backends.cudnn as cudnn
from dataset.kitti.parser import Parser

class AugModel(DensityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_subclusters(self, dataloader, bandwidth=None, max_samples_per_class=8000, sampling_strategy='diverse'):
        """
        Symmetric Phase 1 Subcluster Initialization.
        Uses bundled vectors for the subclusters.
        """
        self.eval()
        num_sub_per_cluster = self.num_subclusters
        print(f"Collecting symmetric bundled embeddings for {self.num_classes} classes")
        all_subcluster_centers = []
        all_subcluster_classes = []

        for class_id in range(self.num_classes):
            print(f"Processing class {class_id}...")
            class_embeddings = []
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    proj_in = batch[0].to(self.device)
                    proj_labels = batch[2].to(self.device).flatten()

                    valid_label_mask = proj_labels >= 0
                    if not valid_label_mask.any():
                        continue
                        
                    proj_labels = proj_labels[valid_label_mask]
                    
                    enc_base, _, _ = self.encode(proj_in)
                    c_base = enc_base[valid_label_mask]
                    del enc_base
                    
                    class_mask = proj_labels == class_id
                    
                    if torch.any(class_mask):
                        bundled = F.normalize(c_base[class_mask])
                        del c_base
                        
                        x_yaw = torch.roll(proj_in, shifts=14, dims=3)
                        enc_yaw, _, _ = self.encode(x_yaw)
                        del x_yaw
                        c_yaw = enc_yaw[valid_label_mask]
                        del enc_yaw
                        bundled.add_(F.normalize(c_yaw[class_mask]))
                        del c_yaw
                        
                        x_scale = proj_in * 0.95
                        enc_scale, _, _ = self.encode(x_scale)
                        del x_scale
                        c_scale = enc_scale[valid_label_mask]
                        del enc_scale
                        bundled.add_(F.normalize(c_scale[class_mask]))
                        del c_scale
                        
                        bundled = F.normalize(bundled).cpu().half()
                        class_embeddings.append(bundled)
                        total_samples += bundled.shape[0]
                    
                    if total_samples >= max_samples_per_class * 2:
                        break
            
            if len(class_embeddings) > 0:
                class_embeddings = torch.cat(class_embeddings, dim=0)
                if class_embeddings.shape[0] > max_samples_per_class:
                    indices = torch.randperm(class_embeddings.shape[0])[:max_samples_per_class]
                    class_embeddings = class_embeddings[indices]
                
                # KNN Subclustering logic (same as HDC_utils)
                from sklearn.cluster import MiniBatchKMeans
                n_clusters = min(num_sub_per_cluster, class_embeddings.shape[0])
                if n_clusters > 0:
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, batch_size=4096)
                    kmeans.fit(class_embeddings.numpy())
                    centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
                    centers = F.normalize(centers, dim=1)
                    all_subcluster_centers.append(centers)
                    all_subcluster_classes.append(torch.full((centers.shape[0],), class_id, dtype=torch.long, device=self.device))
        
        if all_subcluster_centers:
            self.subclusters.data = torch.cat(all_subcluster_centers, dim=0)
            self.subcluster_classes = torch.cat(all_subcluster_classes, dim=0)
            self.subcluster_to_class = self.subcluster_classes
            print(f"Initialized {self.subclusters.shape[0]} subclusters successfully using Symmetric Bundling!")

    def inference_update_symmetric(self, x, learning_rate=0.001, threshold=0.80, **kwargs):
        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            e_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            
            if e_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            e_base = e_base.to(prototypes.dtype)
            bundled_target = e_base
            
            x_yaw = torch.roll(x, shifts=14, dims=3)
            enc_yaw, _, _ = self.encode(x_yaw)
            del x_yaw
            bundled_target.add_(F.normalize(enc_yaw[valid_enc_mask]).to(prototypes.dtype))
            del enc_yaw
            
            x_scale = x * 0.95
            enc_scale, _, _ = self.encode(x_scale)
            del x_scale
            bundled_target.add_(F.normalize(enc_scale[valid_enc_mask]).to(prototypes.dtype))
            del enc_scale
            
            bundled_target = F.normalize(bundled_target)
            
            S = bundled_target @ prototypes.T
            preds = S.argmax(dim=1)
            
            selected_proto = prototypes[preds]
            sims = torch.sum(bundled_target * selected_proto, dim=1)
            
            update_mask = sims > threshold
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                sample_encs = bundled_target[class_mask]
                
                pull_vector = sample_encs.mean(dim=0)
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_asymmetric(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], distance_sensitivity=1.0, proj_xyz=None, oracle_labels=None, **kwargs):
        """Asymmetric Pipeline: Decouples decision from update with combined stability fixes."""
        # Fix D (Drift anchor): capture the source prototypes on the first pass
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()

        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            bundled_target = raw_base.clone()
            
            x_yaw = torch.roll(x, shifts=14, dims=3)
            enc_yaw, _, _ = self.encode(x_yaw)
            del x_yaw
            bundled_target.add_(F.normalize(enc_yaw[valid_enc_mask]).to(prototypes.dtype))
            del enc_yaw
            
            x_scale = x * 0.95
            enc_scale, _, _ = self.encode(x_scale)
            del x_scale
            bundled_target.add_(F.normalize(enc_scale[valid_enc_mask]).to(prototypes.dtype))
            del enc_scale
            
            bundled_target = F.normalize(bundled_target)
            
            # The Decision: Calculate cosine similarity against prototypes using bundled features
            S = bundled_target @ prototypes.T
            preds = S.argmax(dim=1)
            
            selected_proto = prototypes[preds]
            sims = torch.sum(bundled_target * selected_proto, dim=1)
            
            # Fix C (Real DCSP wiring): Build density weights if proj_xyz is available
            density_weights = None
            if proj_xyz is not None:
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                radial_dists = torch.norm(active_xyz, dim=1)
                batch_median = radial_dists.median().clamp(min=1e-5)
                # Upweight sparse/far points relative to batch median density
                density_weights = (radial_dists / batch_median) ** distance_sensitivity
            
            # The Filter: Check if similarities fall within safety thresholds
            if isinstance(thresholds, float):
                thresholds = [thresholds, 1.0]
            update_mask = (sims > thresholds[0]) & (sims > thresholds[0])
            
            # Fix E (Diagnostic logging, opt-in): check pseudo-label accuracy within the band
            if oracle_labels is not None:
                active_oracle = oracle_labels.reshape(-1)[valid_enc_mask]
                if update_mask.sum() > 0:
                    correct = (preds[update_mask] == active_oracle[update_mask]).sum().item()
                    total = update_mask.sum().item()
                    print(f"    [Diagnostic] Band Accuracy: {correct/total:.2%} ({correct}/{total})")
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            # The Update:
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                # Fix B (Purified pull vector): pull with bundled_target instead of raw_base
                sample_encs = bundled_target[class_mask]
                
                # Base similarity to subclusters (distance_sensitivity=1.0 since we scale manually next)
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, 1.0)
                
                if density_weights is not None:
                    sub_sims = sub_sims * density_weights[class_mask]
                
                # Fix A (Volume Normalization Trap)
                sum_evidence = sub_sims.sum()
                normalized_weights = sub_sims / (sum_evidence + 1e-8)
                volume_scale = torch.log1p(sum_evidence)  # scales magnitude by total evidence without blowing up
                
                weighted_pull_vector = (sample_encs * normalized_weights.unsqueeze(1)).sum(dim=0) * volume_scale
                
                # Fix D (Drift anchor): Blend with a small pull back toward the frozen source prototype
                anchor_pull = self.source_prototypes[c_id] - self.classify.weight[c_id]
                anchor_strength = 0.1 * learning_rate  # small constant restoring force
                
                # Apply the composite update to the class prototype
                updated_weight = self.classify.weight[c_id] + learning_rate * weighted_pull_vector + anchor_strength * anchor_pull
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions


    def inference_update_soft_consensus(self, x, learning_rate=0.001, thresholds=[0.35, 0.65],
                                         proj_xyz=None, distance_sensitivity=1.5,
                                         use_consensus_gate=True, use_volume_weight=True,
                                         use_subcluster_gate=True, use_anchor=True, **kwargs):
        """Soft Multi-View Consensus (Experiment A), subcluster-gauged.

        Restores the paper's core mechanism: the confidence used to gate each
        incoming sample is measured against the SUBCLUSTERS (the stored modes
        of the training distribution), not just the averaged class prototype.
        A point that is close to its class prototype but far from every genuine
        training subcluster (a hallmark of a confident-but-wrong corruption
        artifact) is now correctly down-weighted, which class-prototype
        similarity alone could not catch.
        """
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()

        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)

            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)

            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(15.0 / (depth + 1e-3), min=1.0 / distance_sensitivity, max=1.0)

            # Generate augmentations
            x_yaw = torch.roll(x, shifts=14, dims=3)
            enc_yaw, _, _ = self.encode(x_yaw)
            raw_jitter = F.normalize(enc_yaw[valid_enc_mask]).to(prototypes.dtype)
            del x_yaw, enc_yaw

            x_drop = F.dropout2d(x, p=0.15)
            enc_drop, _, _ = self.encode(x_drop)
            raw_drop = F.normalize(enc_drop[valid_enc_mask]).to(prototypes.dtype)
            del x_drop, enc_drop

            # Per-view predictions come from class prototypes (cheap, only used
            # to decide which class each point belongs to, not to gauge confidence).
            pred_base = (raw_base @ prototypes.T).argmax(dim=1)
            if use_consensus_gate:
                pred_jitter = (raw_jitter @ prototypes.T).argmax(dim=1)
                pred_drop = (raw_drop @ prototypes.T).argmax(dim=1)
                agreement_count = (pred_base == pred_jitter).float() + (pred_base == pred_drop).float()
                agreement_weight = agreement_count / 2.0
            else:
                agreement_weight = torch.ones(raw_base.shape[0], device=self.device)

            bundled_target = F.normalize(raw_base + raw_jitter + raw_drop)
            del raw_jitter, raw_drop

            S_bundled = bundled_target @ prototypes.T
            preds = S_bundled.argmax(dim=1)

            # --- THESIS-CRITICAL: confidence gauged against subclusters ---
            # For each point, similarity to the nearest subcluster of its
            # predicted class. This is the "distance to training distribution"
            # gauge the paper is built on. get_max_subcluster_similarity is the
            # same routine the density baseline uses, so the two methods now
            # gate on the same underlying signal.
            if use_subcluster_gate:
                sub_sims = torch.zeros(bundled_target.shape[0], device=self.device, dtype=bundled_target.dtype)
                
                # Vectorized chunked computation to avoid per-class Python loop overhead
                sub_norm = F.normalize(self.subclusters.to(bundled_target.dtype), dim=1)
                chunk_size = 10000
                for i in range(0, bundled_target.shape[0], chunk_size):
                    chunk_target = bundled_target[i:i+chunk_size]
                    chunk_preds = preds[i:i+chunk_size]
                    
                    cosine_sim = chunk_target @ sub_norm.T
                    base_similarity = (cosine_sim + 1) / 2
                    
                    valid_mask = (self.subcluster_to_class.unsqueeze(0) == chunk_preds.unsqueeze(1))
                    masked_similarity = torch.where(valid_mask, base_similarity, torch.tensor(0.0, device=self.device))
                    
                    max_sims, _ = torch.max(masked_similarity, dim=1)
                    
                    sub_sims[i:i+chunk_size] = max_sims.to(sub_sims.dtype)
                    
                gate_sims = sub_sims
            else:
                # fall back to class-prototype similarity (the old behavior)
                gate_sims = S_bundled.gather(1, preds.unsqueeze(1)).squeeze(1)

            update_mask = (gate_sims > thresholds[0]) & (agreement_weight > 0)

            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            if not torch.any(update_mask):
                return full_predictions

            # firing-rate logging (kept outside the gating branches so it's
            # always populated regardless of which components are on)
            firing_rate = update_mask.float().mean().item()
            if not hasattr(self, '_firing_log'):
                self._firing_log = []
            self._firing_log.append(firing_rate)
            
            # Print if firing rate is surprisingly low, to catch scale-mismatches early
            if firing_rate < 0.10:
                print(f"Warning: Firing rate is {firing_rate*100:.2f}%. Gate threshold may be mismatched with similarity scale.")

            unique_classes = torch.unique(preds[torch.nonzero(update_mask).squeeze(1)])
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask

                sample_encs = bundled_target[class_mask]
                sample_gate = gate_sims[class_mask]
                sample_agreement = agreement_weight[class_mask]

                # Confidence weight now ramps over the subcluster-similarity band.
                conf_weights = torch.clamp(
                    (sample_gate - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                    
                # Apply distance sensitivity to the [0,1] confidence weight ramp
                if distance_sensitivity == 0.0:
                    conf_weights = torch.where(conf_weights > 0.5, torch.tensor(1.0, device=self.device), conf_weights * 2.0)
                elif distance_sensitivity != 1.0:
                    conf_weights = conf_weights ** distance_sensitivity
                final_weights = conf_weights * sample_agreement * depth_scale[class_mask]

                if use_volume_weight:
                    weight_sum = final_weights.sum().clamp(min=1e-8)
                    direction = (sample_encs * final_weights.unsqueeze(1)).sum(dim=0) / weight_sum
                    volume_scale = torch.log1p(final_weights.sum())
                    pull_vector = direction * volume_scale
                else:
                    pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)

                if use_anchor:
                    # Drift anchor: Blend with a small pull back toward the frozen source prototype
                    anchor_pull = self.source_prototypes[c_id] - self.classify.weight[c_id]
                    anchor_strength = 0.1 * learning_rate  # small constant restoring force
                    updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector + anchor_strength * anchor_pull
                else:
                    updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_safe_consensus(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Soft Multi-View Consensus with Safety Gating (Safe Exp A)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()
            
        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(15.0 / (depth + 1e-3), min=1.0/distance_sensitivity, max=1.0)

            x_yaw = torch.roll(x, shifts=14, dims=3)
            enc_yaw, _, _ = self.encode(x_yaw)
            raw_jitter = F.normalize(enc_yaw[valid_enc_mask]).to(prototypes.dtype)
            del x_yaw, enc_yaw
            
            x_drop = F.dropout2d(x, p=0.15)
            enc_drop, _, _ = self.encode(x_drop)
            raw_drop = F.normalize(enc_drop[valid_enc_mask]).to(prototypes.dtype)
            del x_drop, enc_drop
            
            # Independent predictions for soft consensus
            S_base = raw_base @ prototypes.T
            pred_base = S_base.argmax(dim=1)
            
            S_jitter = raw_jitter @ prototypes.T
            pred_jitter = S_jitter.argmax(dim=1)
            
            S_drop = raw_drop @ prototypes.T
            pred_drop = S_drop.argmax(dim=1)
            
            # Soft agreement weight: 1.0 if all 3 match, 0.5 if 2 match, 0.0 if none match
            agreement_count = (pred_base == pred_jitter).float() + (pred_base == pred_drop).float()
            agreement_weight = agreement_count / 2.0
            
            bundled_target = F.normalize(raw_base + raw_jitter + raw_drop)
            del raw_jitter, raw_drop
            S_bundled = bundled_target @ prototypes.T
            top2_S = S_bundled.topk(2, dim=1).values
            preds = S_bundled.argmax(dim=1)
            sims = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sims - sims_top2
            
            update_mask = (margin > 0) & (sims > thresholds[0]) & (agreement_weight > 0)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            # --- SAFETY CHECK ---
            num_valid = valid_enc_mask.sum().float()
            firing_rate = update_mask.sum().float() / (num_valid + 1e-6)
            
            if not hasattr(self, '_firing_log'):
                self._firing_log = []
            self._firing_log.append(firing_rate.item())
            
            safe_rate_threshold = 0.20
            if firing_rate > safe_rate_threshold:
                damp_factor = safe_rate_threshold / firing_rate
                learning_rate = learning_rate * damp_factor.item()
            # --------------------
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                # Pull with bundled target (purified), weighted by soft consensus
                sample_encs = bundled_target[class_mask] 
                sample_sims = sims[class_mask]
                sample_agreement = agreement_weight[class_mask]
                
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * sample_agreement * depth_scale[class_mask]
                
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions


    def inference_update_vgp(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, view_variance_scale=10.0, **kwargs):
        """Variance-Gated Purification (VGP)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()

        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)
            
            # Generate augmentations
            x_yaw = torch.roll(x, shifts=14, dims=3)
            enc_yaw, _, _ = self.encode(x_yaw)
            raw_jitter = F.normalize(enc_yaw[valid_enc_mask]).to(prototypes.dtype)
            del x_yaw, enc_yaw
            
            x_drop = F.dropout2d(x, p=0.15)
            enc_drop, _, _ = self.encode(x_drop)
            raw_drop = F.normalize(enc_drop[valid_enc_mask]).to(prototypes.dtype)
            del x_drop, enc_drop
            
            # Independent predictions for soft consensus
            S_base = raw_base @ prototypes.T
            pred_base = S_base.argmax(dim=1)
            sim_base = S_base.max(dim=1).values
            
            S_jitter = raw_jitter @ prototypes.T
            pred_jitter = S_jitter.argmax(dim=1)
            sim_jitter = S_jitter.max(dim=1).values
            
            S_drop = raw_drop @ prototypes.T
            pred_drop = S_drop.argmax(dim=1)
            sim_drop = S_drop.max(dim=1).values
            
            # Soft agreement weight: 1.0 if all 3 match, 0.5 if 2 match, 0.0 if none match
            agreement_count = (pred_base == pred_jitter).float() + (pred_base == pred_drop).float()
            agreement_weight = agreement_count / 2.0

            # VGP: variance gating based on max similarities across views
            sims_per_view = torch.stack([sim_base, sim_jitter, sim_drop])
            variance_weight = torch.exp(-view_variance_scale * sims_per_view.var(dim=0))
            
            bundled_target = F.normalize(raw_base + raw_jitter + raw_drop)
            del raw_jitter, raw_drop
            S_bundled = bundled_target @ prototypes.T
            top2_S = S_bundled.topk(2, dim=1).values
            preds = S_bundled.argmax(dim=1)
            sims = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sims - sims_top2
            
            update_mask = (margin > 0) & (sims > thresholds[0]) & (agreement_weight > 0)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                sample_encs = bundled_target[class_mask] 
                sample_sims = sims[class_mask]
                sample_agreement = agreement_weight[class_mask]
                sample_var_weight = variance_weight[class_mask]
                
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * sample_agreement * sample_var_weight * depth_scale[class_mask]
                
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_soft_dcsp(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Soft Multi-View Consensus + Density Weighting (Experiment B)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()

        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            
            x_yaw = torch.roll(x, shifts=14, dims=3)
            enc_yaw, _, _ = self.encode(x_yaw)
            raw_jitter = F.normalize(enc_yaw[valid_enc_mask]).to(prototypes.dtype)
            del x_yaw, enc_yaw
            
            x_drop = F.dropout2d(x, p=0.15)
            enc_drop, _, _ = self.encode(x_drop)
            raw_drop = F.normalize(enc_drop[valid_enc_mask]).to(prototypes.dtype)
            del x_drop, enc_drop
            
            S_base = raw_base @ prototypes.T
            pred_base = S_base.argmax(dim=1)
            
            S_jitter = raw_jitter @ prototypes.T
            pred_jitter = S_jitter.argmax(dim=1)
            
            S_drop = raw_drop @ prototypes.T
            pred_drop = S_drop.argmax(dim=1)
            
            agreement_count = (pred_base == pred_jitter).float() + (pred_base == pred_drop).float()
            agreement_weight = agreement_count / 2.0
            
            bundled_target = F.normalize(raw_base + raw_jitter + raw_drop)
            del raw_jitter, raw_drop
            S_bundled = bundled_target @ prototypes.T
            preds = S_bundled.argmax(dim=1)
            sims = S_bundled.max(dim=1).values
            
            density_weights = None
            if proj_xyz is not None:
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                radial_dists = torch.norm(active_xyz, dim=1)
                batch_median = radial_dists.median().clamp(min=1e-5)
                density_weights = (radial_dists / batch_median).clamp(max=1.5)
            
            update_mask = (sims > thresholds[0]) & (sims > thresholds[0]) & (agreement_weight > 0)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                sample_encs = bundled_target[class_mask] 
                sample_agreement = agreement_weight[class_mask]
                
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, 1.0)
                
                if density_weights is not None:
                    sub_sims = sub_sims * sample_agreement * density_weights[class_mask]
                else:
                    sub_sims = sub_sims * sample_agreement
                    
                pull_vector = (sample_encs * sub_sims.unsqueeze(1)).mean(dim=0)
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_mssb(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Multi-Scale Spatial Bundling (MSSB)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()
            
        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            
            x_coarse = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            enc_coarse, _, _ = self.encode(x_coarse)
            raw_coarse = F.normalize(enc_coarse[valid_enc_mask]).to(prototypes.dtype)
            del x_coarse, enc_coarse
            
            bundled_target = F.normalize(raw_base + raw_coarse)
            S = bundled_target @ prototypes.T
            top2_S = S.topk(2, dim=1).values
            preds = S.argmax(dim=1)
            sims = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sims - sims_top2
            
            update_mask = (margin > 0.08) & (sims > thresholds[0])
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                sample_encs = bundled_target[class_mask] 
                sample_sims = sims[class_mask]
                
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * depth_scale[class_mask]
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_opp(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Orthogonalized Prototype Pull (OPP) - Plain"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()
            
        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            
            S = raw_base @ prototypes.T
            top2_S = S.topk(2, dim=1).values
            preds = S.argmax(dim=1)
            sims = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sims - sims_top2
            
            update_mask = (margin > 0.08) & (sims > thresholds[0])
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                sample_encs = raw_base[class_mask] 
                sample_sims = sims[class_mask]
                
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * depth_scale[class_mask]
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                other_protos = prototypes[torch.arange(prototypes.shape[0]) != c_id]
                sims_to_others = other_protos @ F.normalize(pull_vector, dim=0)
                nearest_k = sims_to_others.topk(min(3, other_protos.shape[0])).indices
                
                for k in nearest_k:
                    component = torch.dot(pull_vector, other_protos[k]) * other_protos[k]
                    pull_vector = pull_vector - component.clamp(min=0) * 1.0
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_conditional_opp(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Conditional Orthogonalized Prototype Pull (C-OPP)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()
            
        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            
            S = raw_base @ prototypes.T
            top2_S = S.topk(2, dim=1).values
            preds = S.argmax(dim=1)
            sims = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sims - sims_top2
            
            update_mask = (margin > 0.08) & (sims > thresholds[0])
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                
                sample_encs = raw_base[class_mask] 
                sample_sims = sims[class_mask]
                
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * depth_scale[class_mask]
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                other_protos = prototypes[torch.arange(prototypes.shape[0]) != c_id]
                target_proto = prototypes[c_id]
                proto_sims = other_protos @ target_proto
                
                if proto_sims.max() > 0.70:
                    sims_to_others = other_protos @ F.normalize(pull_vector, dim=0)
                    nearest_k = sims_to_others.topk(min(3, other_protos.shape[0])).indices
                    
                    for k in nearest_k:
                        component = torch.dot(pull_vector, other_protos[k]) * other_protos[k]
                        pull_vector = pull_vector - component.clamp(min=0) * 1.0
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_cwsa(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Coverage-Weighted Subcluster Allocation (CWSA)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()
            
        if not hasattr(self, 'subcluster_activation_counts') or self.subcluster_activation_counts.shape[0] != self.subclusters.shape[0]:
            self.subcluster_activation_counts = torch.zeros(self.subclusters.shape[0], device=self.device)

        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            
            S = raw_base @ prototypes.T
            top2_S = S.topk(2, dim=1).values
            preds = S.argmax(dim=1)
            sims = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sims - sims_top2
            
            update_mask = (margin > 0.08) & (sims > thresholds[0])
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                sample_encs = raw_base[class_mask] 
                
                class_subcluster_ids = torch.nonzero(self.subcluster_to_class == c_id).squeeze(1)
                if len(class_subcluster_ids) == 0:
                    continue
                    
                all_sub_sims = sample_encs @ F.normalize(self.subclusters[class_subcluster_ids]).T
                k_to_use = min(3, all_sub_sims.shape[1])
                topk_sims, topk_idx = all_sub_sims.topk(k_to_use, dim=1)
                
                global_topk_idx = class_subcluster_ids[topk_idx]
                
                starvation_weight = 1.0 / (self.subcluster_activation_counts[global_topk_idx] + 10.0)
                combined_weight = topk_sims.clamp(min=0) * starvation_weight
                
                unique_idx, counts = torch.unique(global_topk_idx, return_counts=True)
                self.subcluster_activation_counts[unique_idx] += counts.to(self.subcluster_activation_counts.dtype)
                        
                sub_sims = combined_weight.sum(dim=1)
                
                sample_sims = sims[class_mask]
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = sub_sims * conf_weights * depth_scale[class_mask]
                
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_ltcg(self, x, learning_rate=0.001, thresholds=[0.35, 0.65], proj_xyz=None, distance_sensitivity=1.5, **kwargs):
        """Low-Threshold Consensus Gating (LTCG)"""
        if not hasattr(self, 'source_prototypes'):
            self.source_prototypes = self.classify.weight.detach().clone()
            
        with torch.no_grad():
            enc_base, _, _ = self.encode(x)
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            raw_base = F.normalize(enc_base[valid_enc_mask])
            del enc_base
            if raw_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            raw_base = raw_base.to(prototypes.dtype)

            depth_scale = torch.ones(raw_base.shape[0], device=self.device)
            if proj_xyz is not None and distance_sensitivity > 0:
                flat_xyz = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)[valid_enc_mask]
                depth = torch.norm(flat_xyz, dim=1)
                depth_scale = torch.clamp(depth / 10.0, min=1.0, max=distance_sensitivity)

            
            S_base = raw_base @ prototypes.T
            top2_S = S_base.topk(2, dim=1).values
            pred_base = S_base.argmax(dim=1)
            sim_base = top2_S[:, 0]
            sims_top2 = top2_S[:, 1]
            margin = sim_base - sims_top2
            
            # Yaw Jitter
            x_jitter = x.roll(shifts=3, dims=3)
            enc_jitter, _, _ = self.encode(x_jitter)
            raw_jitter = F.normalize(enc_jitter[valid_enc_mask]).to(prototypes.dtype)
            del x_jitter, enc_jitter
            pred_jitter = (raw_jitter @ prototypes.T).argmax(dim=1)
            
            # Spatial Dropout
            dropout_mask = torch.rand(x.shape, device=x.device) > 0.1
            x_drop = x * dropout_mask
            enc_drop, _, _ = self.encode(x_drop)
            raw_drop = F.normalize(enc_drop[valid_enc_mask]).to(prototypes.dtype)
            del x_drop, enc_drop
            pred_drop = (raw_drop @ prototypes.T).argmax(dim=1)
            del raw_jitter, raw_drop
            
            # LTCG Logic
            # Require consensus from all 3 views AND a lower similarity threshold
            update_mask = (margin > 0.04) & (sim_base > thresholds[0]) & (pred_base == pred_jitter) & (pred_base == pred_drop)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = pred_base
            
            if not torch.any(update_mask):
                return full_predictions
                
            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(pred_base[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (pred_base == c_id) & update_mask
                
                # Apply Standard Pull using ONLY the raw H_base
                sample_encs = raw_base[class_mask] 
                sample_sims = sim_base[class_mask]
                
                conf_weights = torch.clamp((sample_sims - thresholds[0]) / (thresholds[1] - thresholds[0]), 0.0, 1.0)
                final_weights = conf_weights * depth_scale[class_mask]
                pull_vector = (sample_encs * final_weights.unsqueeze(1)).mean(dim=0)
                
                updated_weight = self.classify.weight[c_id] + learning_rate * pull_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

class AugTrainer(DensityTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the model created by DensityTrainer
        self.model = AugModel(self.ARCH, self.modeldir, 'rp', 0, 0, self.num_classes, self.device, subcluster_type='continuous')
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.model.cuda()

    def train(self, train_loader, model, logger):
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            self.is_wrong_list = [None] * len(train_loader)
            for i, batch in enumerate(tqdm(train_loader, desc="Training (Symmetric)")):
                proj_in = batch[0]
                proj_labels = batch[2]
                
                if self.gpu:
                    proj_in = proj_in.cuda()
                    
                enc_base, _, _ = self.model.encode(proj_in)
                
                valid_mask = (enc_base.abs().sum(dim=1) > 0)
                num_total_samples = enc_base.shape[0]
                hd_dim = enc_base.shape[1]
                
                bundled = F.normalize(enc_base[valid_mask])
                del enc_base
                
                x_yaw = torch.roll(proj_in, shifts=14, dims=3)
                enc_yaw, _, _ = self.model.encode(x_yaw)
                del x_yaw
                bundled.add_(F.normalize(enc_yaw[valid_mask]))
                del enc_yaw
                
                x_scale = proj_in * 0.95
                enc_scale, _, _ = self.model.encode(x_scale)
                del x_scale
                bundled.add_(F.normalize(enc_scale[valid_mask]))
                del enc_scale
                
                bundled = F.normalize(bundled).to(model.classify_weights.dtype)
                samples_hv = torch.zeros((num_total_samples, hd_dim), device=self.device, dtype=model.classify_weights.dtype)
                samples_hv[valid_mask] = bundled
                del bundled
                
                proj_labels = proj_labels.view(-1).to(self.device)
                
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                
                predictions = self.model.get_predictions(samples_hv)
                argmax = predictions.argmax(dim=1)
                
                is_wrong = proj_labels != argmax
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                
                true_scores = predictions[is_wrong, proj_labels]
                wrong_scores = predictions[is_wrong, argmax]
                losses = wrong_scores - true_scores
                
                self.is_wrong_list[i] = is_wrong

            if self.bipolar_prototypes:
                with torch.no_grad():
                    model.classify_weights.data = torch.sign(model.classify_weights.data)
                    zero_mask = model.classify_weights.data == 0
                    if torch.any(zero_mask):
                        model.classify_weights.data[zero_mask] = -1.0
                    model.classify.weight.data = model.classify_weights.data.clone()
            else:
                model.classify.weight[:] = F.normalize(model.classify_weights)

    def retrain(self, train_loader, model, epoch, logger):
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader, desc=f"Retraining Epoch {epoch}")):
                proj_in = batch[0]
                proj_labels = batch[2]
                
                if self.gpu:
                    proj_in = proj_in.cuda()
                    
                enc_base, _, _ = self.model.encode(proj_in)
                
                valid_mask = (enc_base.abs().sum(dim=1) > 0)
                num_total_samples = enc_base.shape[0]
                hd_dim = enc_base.shape[1]
                
                bundled = F.normalize(enc_base[valid_mask])
                del enc_base
                
                x_yaw = torch.roll(proj_in, shifts=14, dims=3)
                enc_yaw, _, _ = self.model.encode(x_yaw)
                del x_yaw
                bundled.add_(F.normalize(enc_yaw[valid_mask]))
                del enc_yaw
                
                x_scale = proj_in * 0.95
                enc_scale, _, _ = self.model.encode(x_scale)
                del x_scale
                bundled.add_(F.normalize(enc_scale[valid_mask]))
                del enc_scale
                
                bundled = F.normalize(bundled).to(model.classify_weights.dtype)
                samples_hv = torch.zeros((num_total_samples, hd_dim), device=self.device, dtype=model.classify_weights.dtype)
                samples_hv[valid_mask] = bundled
                del bundled
                
                proj_labels = proj_labels.view(-1).to(self.device)
                
                predictions = self.model.get_predictions(samples_hv)
                argmax = predictions.argmax(dim=1)
                
                is_wrong = proj_labels != argmax
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                
                true_scores = predictions[is_wrong, proj_labels]
                wrong_scores = predictions[is_wrong, argmax]
                losses = wrong_scores - true_scores
                
                lr = max(0.001, 1.0 - (epoch / self.epochs))
                
                if samples_hv.shape[0] > 0:
                    model.classify_weights.index_add_(0, proj_labels, samples_hv, alpha=lr)
                    model.classify_weights.index_add_(0, argmax, samples_hv, alpha=-lr)
                    
            if self.bipolar_prototypes:
                with torch.no_grad():
                    model.classify_weights.data = torch.sign(model.classify_weights.data)
                    zero_mask = model.classify_weights.data == 0
                    if torch.any(zero_mask):
                        model.classify_weights.data[zero_mask] = -1.0
                    model.classify.weight.data = model.classify_weights.data.clone()
            else:
                model.classify.weight[:] = F.normalize(model.classify_weights)


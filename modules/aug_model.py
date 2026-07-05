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

    def inference_update_asymmetric(self, x, learning_rate=0.001, thresholds=[0.45, 0.80], distance_sensitivity=1.0, proj_xyz=None, oracle_labels=None, **kwargs):
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
            update_mask = (sims > thresholds[0]) & (sims < thresholds[1])
            
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

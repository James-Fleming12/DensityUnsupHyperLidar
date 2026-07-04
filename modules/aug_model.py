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
                    
                    x_yaw = torch.roll(proj_in, shifts=14, dims=3)
                    x_scale = proj_in * 0.95
                    
                    enc_base, _, _ = self.encode(proj_in)
                    enc_yaw, _, _ = self.encode(x_yaw)
                    enc_scale, _, _ = self.encode(x_scale)
                    
                    enc_base = enc_base[valid_label_mask]
                    enc_yaw = enc_yaw[valid_label_mask]
                    enc_scale = enc_scale[valid_label_mask]

                    class_mask = proj_labels == class_id
                    
                    if torch.any(class_mask):
                        c_base = F.normalize(enc_base[class_mask])
                        c_yaw = F.normalize(enc_yaw[class_mask])
                        c_scale = F.normalize(enc_scale[class_mask])
                        
                        bundled = F.normalize(c_base + c_yaw + c_scale).cpu().half()
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
            x_yaw = torch.roll(x, shifts=14, dims=3)
            x_scale = x * 0.95
            
            enc_base, _, _ = self.encode(x)
            enc_yaw, _, _ = self.encode(x_yaw)
            enc_scale, _, _ = self.encode(x_scale)
            
            num_total_samples = enc_base.shape[0]
            valid_enc_mask = (enc_base.abs().sum(dim=1) > 0)
            
            e_base = F.normalize(enc_base[valid_enc_mask])
            e_yaw = F.normalize(enc_yaw[valid_enc_mask])
            e_scale = F.normalize(enc_scale[valid_enc_mask])
            
            if e_base.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if e_base.dtype != prototypes.dtype:
                e_base = e_base.to(prototypes.dtype)
                e_yaw = e_yaw.to(prototypes.dtype)
                e_scale = e_scale.to(prototypes.dtype)
                
            bundled_target = F.normalize(e_base + e_yaw + e_scale)
            
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
                    
                x_yaw = torch.roll(proj_in, shifts=14, dims=3)
                x_scale = proj_in * 0.95
                
                enc_base, _, _ = self.model.encode(proj_in)
                enc_yaw, _, _ = self.model.encode(x_yaw)
                enc_scale, _, _ = self.model.encode(x_scale)
                
                # We need to construct samples_hv as the bundled version
                samples_hv = torch.zeros_like(enc_base, dtype=model.classify_weights.dtype)
                valid_mask = (enc_base.abs().sum(dim=1) > 0)
                
                e_base = F.normalize(enc_base[valid_mask])
                e_yaw = F.normalize(enc_yaw[valid_mask])
                e_scale = F.normalize(enc_scale[valid_mask])
                
                bundled = F.normalize(e_base + e_yaw + e_scale).to(model.classify_weights.dtype)
                samples_hv[valid_mask] = bundled
                
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
                    
                x_yaw = torch.roll(proj_in, shifts=14, dims=3)
                x_scale = proj_in * 0.95
                
                enc_base, _, _ = self.model.encode(proj_in)
                enc_yaw, _, _ = self.model.encode(x_yaw)
                enc_scale, _, _ = self.model.encode(x_scale)
                
                samples_hv = torch.zeros_like(enc_base, dtype=model.classify_weights.dtype)
                valid_mask = (enc_base.abs().sum(dim=1) > 0)
                
                e_base = F.normalize(enc_base[valid_mask])
                e_yaw = F.normalize(enc_yaw[valid_mask])
                e_scale = F.normalize(enc_scale[valid_mask])
                
                bundled = F.normalize(e_base + e_yaw + e_scale).to(model.classify_weights.dtype)
                samples_hv[valid_mask] = bundled
                
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

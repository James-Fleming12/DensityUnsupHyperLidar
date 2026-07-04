import torch
import torch.nn.functional as F

def inference_update_cws(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, gamma=2.0, max_updates_per_class=-1, **kwargs):
    with torch.no_grad():
        enc, _, _ = self.encode(x)
        num_total_samples = enc.shape[0]
        valid_enc_mask = (enc.abs().sum(dim=1) > 0)
        enc_norm = F.normalize(enc[valid_enc_mask])
        
        if enc_norm.shape[0] == 0:
            return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
        prototypes = F.normalize(self.classify.weight)
        if enc_norm.dtype != prototypes.dtype:
            enc_norm = enc_norm.to(prototypes.dtype)
        
        S = enc_norm @ prototypes.T
        preds = S.argmax(dim=1)
        
        x_aug1 = torch.roll(x, shifts=14, dims=3)
        x_aug2 = x * 1.05
        
        enc1, _, _ = self.encode(x_aug1)
        enc2, _, _ = self.encode(x_aug2)
        
        enc1_norm = F.normalize(enc1[valid_enc_mask])
        enc2_norm = F.normalize(enc2[valid_enc_mask])
        
        if enc1_norm.dtype != prototypes.dtype:
            enc1_norm = enc1_norm.to(prototypes.dtype)
        if enc2_norm.dtype != prototypes.dtype:
            enc2_norm = enc2_norm.to(prototypes.dtype)
            
        selected_proto = prototypes[preds]
        
        w0 = torch.sum(enc_norm * selected_proto, dim=1).clamp(min=0.0)
        w1 = torch.sum(enc1_norm * selected_proto, dim=1).clamp(min=0.0)
        w2 = torch.sum(enc2_norm * selected_proto, dim=1).clamp(min=0.0)
        
        w0 = (w0 ** gamma).unsqueeze(1)
        w1 = (w1 ** gamma).unsqueeze(1)
        w2 = (w2 ** gamma).unsqueeze(1)
        
        bundled_enc = F.normalize((w0 * enc_norm) + (w1 * enc1_norm) + (w2 * enc2_norm))
        
        sims = torch.sum(bundled_enc * selected_proto, dim=1)
        distances = (1.0 - sims) / 2.0
        update_mask = distances > beta
        
        full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        full_predictions[valid_enc_mask] = preds
        
        if not torch.any(update_mask):
            return full_predictions

        valid_indices = torch.nonzero(update_mask).squeeze(1)
        unique_classes = torch.unique(preds[valid_indices])
        
        for class_id in unique_classes:
            c_id = class_id.item()
            class_mask = (preds == c_id) & update_mask
            sample_encs = bundled_enc[class_mask]
            
            pull_vector = sample_encs.mean(dim=0)
            self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
            updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
            self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
            
        return full_predictions

def inference_update_dava(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, **kwargs):
    with torch.no_grad():
        enc, _, _ = self.encode(x)
        num_total_samples = enc.shape[0]
        valid_enc_mask = (enc.abs().sum(dim=1) > 0)
        enc_norm = F.normalize(enc[valid_enc_mask])
        
        if enc_norm.shape[0] == 0:
            return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
        prototypes = F.normalize(self.classify.weight)
        if enc_norm.dtype != prototypes.dtype:
            enc_norm = enc_norm.to(prototypes.dtype)
            
        S = enc_norm @ prototypes.T
        preds = S.argmax(dim=1)
        
        bundled_enc = enc_norm.clone()
        
        if proj_xyz is not None:
            xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
            active_xyz = xyz_flat[valid_enc_mask]
            
            chunk_s = 5000
            num_active = active_xyz.shape[0]
            densities = torch.zeros(num_active, device=self.device)
            for i in range(0, num_active, chunk_s):
                end = min(i + chunk_s, num_active)
                chunk_xyz = active_xyz[i:end]
                dist = torch.cdist(chunk_xyz, active_xyz)
                densities[i:end] = (dist < 0.5).sum(dim=1).float()
            
            density_threshold = 5 
            is_sparse = densities < density_threshold
            is_dense = ~is_sparse
            
            if torch.any(is_dense):
                x_yaw = torch.roll(x, shifts=14, dims=3)
                enc_yaw, _, _ = self.encode(x_yaw)
                enc_yaw_norm = F.normalize(enc_yaw[valid_enc_mask][is_dense])
                if enc_yaw_norm.dtype != prototypes.dtype:
                    enc_yaw_norm = enc_yaw_norm.to(prototypes.dtype)
                bundled_enc[is_dense] = F.normalize(enc_norm[is_dense] + enc_yaw_norm)
            
            if torch.any(is_sparse):
                x_yaw = torch.roll(x, shifts=14, dims=3)
                x_scale = x * 1.05
                x_jitter = x + torch.randn_like(x) * 0.01
                x_drop = F.dropout(x, p=0.1)
                
                enc_yaw, _, _ = self.encode(x_yaw)
                enc_scale, _, _ = self.encode(x_scale)
                enc_jitter, _, _ = self.encode(x_jitter)
                enc_drop, _, _ = self.encode(x_drop)
                
                e_yaw = F.normalize(enc_yaw[valid_enc_mask][is_sparse])
                e_scale = F.normalize(enc_scale[valid_enc_mask][is_sparse])
                e_jitter = F.normalize(enc_jitter[valid_enc_mask][is_sparse])
                e_drop = F.normalize(enc_drop[valid_enc_mask][is_sparse])
                
                for e in [e_yaw, e_scale, e_jitter, e_drop]:
                    if e.dtype != prototypes.dtype:
                        e.copy_(e.to(prototypes.dtype))
                
                bundled_enc[is_sparse] = F.normalize(
                    enc_norm[is_sparse] + 
                    e_yaw.to(prototypes.dtype) + 
                    e_scale.to(prototypes.dtype) + 
                    e_jitter.to(prototypes.dtype) + 
                    e_drop.to(prototypes.dtype)
                )
        
        selected_proto = prototypes[preds]
        sims = torch.sum(bundled_enc * selected_proto, dim=1)
        distances = (1.0 - sims) / 2.0
        update_mask = distances > beta
        
        full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        full_predictions[valid_enc_mask] = preds
        
        if not torch.any(update_mask):
            return full_predictions

        valid_indices = torch.nonzero(update_mask).squeeze(1)
        unique_classes = torch.unique(preds[valid_indices])
        
        for class_id in unique_classes:
            c_id = class_id.item()
            class_mask = (preds == c_id) & update_mask
            sample_encs = bundled_enc[class_mask]
            
            pull_vector = sample_encs.mean(dim=0)
            self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
            updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
            self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
            
        return full_predictions

def inference_update_mssb(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, **kwargs):
    with torch.no_grad():
        enc, _, _ = self.encode(x)
        num_total_samples = enc.shape[0]
        valid_enc_mask = (enc.abs().sum(dim=1) > 0)
        enc_norm = F.normalize(enc[valid_enc_mask])
        
        if enc_norm.shape[0] == 0:
            return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
        prototypes = F.normalize(self.classify.weight)
        if enc_norm.dtype != prototypes.dtype:
            enc_norm = enc_norm.to(prototypes.dtype)
            
        S = enc_norm @ prototypes.T
        preds = S.argmax(dim=1)
        
        x_small = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_large = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        
        enc_small, _, _ = self.encode(x_small)
        enc_large, _, _ = self.encode(x_large)
        
        enc_small_norm = F.normalize(enc_small[valid_enc_mask])
        enc_large_norm = F.normalize(enc_large[valid_enc_mask])
        
        if enc_small_norm.dtype != prototypes.dtype:
            enc_small_norm = enc_small_norm.to(prototypes.dtype)
        if enc_large_norm.dtype != prototypes.dtype:
            enc_large_norm = enc_large_norm.to(prototypes.dtype)
            
        bundled_enc = F.normalize(enc_norm + enc_small_norm + enc_large_norm)
        
        selected_proto = prototypes[preds]
        sims = torch.sum(bundled_enc * selected_proto, dim=1)
        distances = (1.0 - sims) / 2.0
        update_mask = distances > beta
        
        full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        full_predictions[valid_enc_mask] = preds
        
        if not torch.any(update_mask):
            return full_predictions

        valid_indices = torch.nonzero(update_mask).squeeze(1)
        unique_classes = torch.unique(preds[valid_indices])
        
        for class_id in unique_classes:
            c_id = class_id.item()
            class_mask = (preds == c_id) & update_mask
            sample_encs = bundled_enc[class_mask]
            
            pull_vector = sample_encs.mean(dim=0)
            self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
            updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
            self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
            
        return full_predictions

def inference_update_tha(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, alpha=0.5, **kwargs):
    with torch.no_grad():
        enc, _, _ = self.encode(x)
        num_total_samples = enc.shape[0]
        valid_enc_mask = (enc.abs().sum(dim=1) > 0)
        enc_norm = F.normalize(enc[valid_enc_mask])
        
        if enc_norm.shape[0] == 0:
            return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
        prototypes = F.normalize(self.classify.weight)
        if enc_norm.dtype != prototypes.dtype:
            enc_norm = enc_norm.to(prototypes.dtype)
            
        if not hasattr(self, 'tha_memory'):
            self.tha_memory = torch.zeros((num_total_samples, self.hd_dim), device=self.device, dtype=prototypes.dtype)
            
        self.tha_memory[valid_enc_mask] = F.normalize(
            alpha * self.tha_memory[valid_enc_mask] + (1 - alpha) * enc_norm
        )
        
        bundled_enc = self.tha_memory[valid_enc_mask]
        
        S = bundled_enc @ prototypes.T
        preds = S.argmax(dim=1)
        
        selected_proto = prototypes[preds]
        sims = torch.sum(bundled_enc * selected_proto, dim=1)
        distances = (1.0 - sims) / 2.0
        update_mask = distances > beta
        
        full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        full_predictions[valid_enc_mask] = preds
        
        if not torch.any(update_mask):
            return full_predictions

        valid_indices = torch.nonzero(update_mask).squeeze(1)
        unique_classes = torch.unique(preds[valid_indices])
        
        for class_id in unique_classes:
            c_id = class_id.item()
            class_mask = (preds == c_id) & update_mask
            sample_encs = bundled_enc[class_mask]
            
            pull_vector = sample_encs.mean(dim=0)
            self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
            updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
            self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
            
        return full_predictions

with open('/home/james/Research/SEE/DensityUnsupHyperLidar/modules/active_model.py', 'a') as f:
    f.write("\n    inference_update_cws = inference_update_cws\n")
    f.write("    inference_update_dava = inference_update_dava\n")
    f.write("    inference_update_mssb = inference_update_mssb\n")
    f.write("    inference_update_tha = inference_update_tha\n")

print("Methods appended successfully!")

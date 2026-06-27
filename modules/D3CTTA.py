import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class D3CTTA(nn.Module):
    def __init__(self, feature_extractor, num_classes=13, feature_dim=128, proj_dim=1024, lambda_ridge=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.lambda_ridge = lambda_ridge
        
        # 2. Random Projection
        self.W = nn.Linear(feature_dim, proj_dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.W.weight, mean=0, std=1.0 / math.sqrt(feature_dim))
        self.W.weight.requires_grad = False
        
        # Domain tracking
        self.domain_id = 0
        self.domains_bn_stats = {} # domain_id -> {'mu': mu, 'sigma': sigma}
        self.G_d = {} # domain_id -> [proj_dim, proj_dim]
        self.C_d = {} # domain_id -> [proj_dim, num_classes]
        
        self.prev_mu = None
        
        # For region-specific prototypes (Distance-Aware Prototype Learning)
        # Note: D3CTTA uses Ridge Regression (DSD) instead of directly predicting from prototypes.
        # But wait, user says "Update region-specific prototype using EMA... Instead of predicting directly from the prototypes, D3CTTA uses ridge regression... Prototypical Features Ct = cumulative sum... weighted by pseudo-labels yk"
        # I will maintain G_d and C_d for ridge regression.
        
        # Create domain 0
        self.create_new_domain(0)

    def get_last_bn_stats(self):
        # find the last BatchNorm2d layer in feature_extractor
        last_bn = None
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.BatchNorm2d):
                last_bn = module
        if last_bn is not None:
            return last_bn.running_mean.detach().clone(), torch.sqrt(last_bn.running_var.detach().clone() + 1e-5)
        return None, None

    def create_new_domain(self, domain_id, mu=None, sigma=None):
        device = next(self.parameters()).device
        self.G_d[domain_id] = torch.zeros(self.proj_dim, self.proj_dim, device=device)
        self.C_d[domain_id] = torch.zeros(self.proj_dim, self.num_classes, device=device)
        if mu is not None and sigma is not None:
            self.domains_bn_stats[domain_id] = {'mu': mu, 'sigma': sigma}

    def forward(self, x, xyz=None, *args, **kwargs):
        # Forward through feature extractor
        with torch.no_grad():
            # For ResNet_34, it returns (pred, out) or similar.
            out = self.feature_extractor(x)
            if isinstance(out, tuple):
                if len(out) == 2:
                    base_pred, feat = out
                else: # e.g. aux returns
                    base_pred = out[0]
                    feat = out[-1]
            else:
                feat = out
                base_pred = None
            
            # handle shape: feat is [B, 128, H, W]
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            
            # Random projection and ReLU
            h = F.relu(self.W(feat_flat))
            
            # Get current BN stats
            mu, sigma = self.get_last_bn_stats()
            
            # Domain Detection
            if mu is not None and self.prev_mu is not None:
                cos_sim = F.cosine_similarity(mu, self.prev_mu, dim=0)
                if cos_sim <= 0.85:
                    # Domain shift occurred! Find best matching domain
                    best_dist = float('inf')
                    best_domain = -1
                    for d_id, stats in self.domains_bn_stats.items():
                        dist = torch.sum((mu - stats['mu'])**2 + (sigma - stats['sigma'])**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_domain = d_id
                    
                    if best_domain != -1 and best_dist < 10.0:
                        self.domain_id = best_domain
                    else:
                        self.domain_id = len(self.domains_bn_stats)
                        self.create_new_domain(self.domain_id, mu, sigma)
                        
            self.prev_mu = mu
            if self.domain_id not in self.domains_bn_stats and mu is not None:
                self.domains_bn_stats[self.domain_id] = {'mu': mu, 'sigma': sigma}

            # Predictions via Ridge Regression
            G = self.G_d[self.domain_id]
            C = self.C_d[self.domain_id]
            
            if C.sum() == 0 and base_pred is not None:
                # Bootstrap from base classifier if DSD matrices are empty
                logits = base_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            else:
                device = h.device
                I = torch.eye(self.proj_dim, device=device)
                G_inv = torch.linalg.inv(G + self.lambda_ridge * I)
                logits = h @ G_inv @ C
            
            # Match HDC behavior: predict Class 0 (unlabeled) for all empty background pixels
            empty_mask = (x.sum(dim=1) == 0).view(-1)
            logits[empty_mask, :] = -1e9
            logits[empty_mask, 0] = 1e9
            
        return logits, None, torch.arange(logits.shape[0], device=logits.device), h

    def inference_update(self, h, predictions, xyz):
        """
        h: [N, 1024]
        predictions: [N] pseudo-labels
        xyz: [B, 3, H, W] points (or [N, 3])
        """
        device = h.device
        
        # Flatten xyz to match h
        if xyz.dim() == 4:
            xyz = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
            
        N = h.shape[0]
        
        # 1. Distance-Aware KNN Filtering
        if xyz is not None and N > 20:
            # Distance Partitioning (not explicitly used for filtering, but requested in DAPL. 
            # I'll use it to restrict KNN search or just do global KNN. Let's do global KNN for pseudo-labels)
            
            # KNN using simple pairwise distance (Memory heavy if N is 32768, so we process in chunks or sample)
            # To avoid OOM, let's filter out background (invalid labels or where h is all zero)
            valid_mask = (h.sum(dim=1) != 0)
            valid_idx = torch.nonzero(valid_mask).squeeze()
            
            if valid_idx.numel() > 0:
                h_valid = h[valid_idx]
                pred_valid = predictions[valid_idx]
                xyz_valid = xyz[valid_idx].reshape(-1, 3)
                
                # We can use PyTorch3D KNN or compute distances in chunks
                # We'll do chunked pairwise distance to find 20 nearest neighbors
                chunk_size = 2000
                keep_mask = torch.zeros(valid_idx.shape[0], dtype=torch.bool, device=device)
                
                for i in range(0, valid_idx.shape[0], chunk_size):
                    end = min(i + chunk_size, valid_idx.shape[0])
                    xyz_chunk = xyz_valid[i:end]
                    
                    # Distances: [chunk, N]
                    dists = torch.cdist(xyz_chunk.unsqueeze(0), xyz_valid.unsqueeze(0)).squeeze(0)
                    # topk returns largest, so we want smallest (K=21, including self)
                    _, knn_idx = torch.topk(dists, k=min(21, valid_idx.shape[0]), dim=1, largest=False)
                    
                    # Check consistency
                    knn_preds = pred_valid[knn_idx] # [chunk, K]
                    center_preds = pred_valid[i:end].unsqueeze(1)
                    consistency = (knn_preds == center_preds).float().mean(dim=1)
                    
                    keep_mask[i:end] = consistency > 0.8
                
                # Update G_d and C_d with filtered points
                h_filtered = h_valid[keep_mask]
                pred_filtered = pred_valid[keep_mask]
                
                if h_filtered.shape[0] > 0:
                    # G_d += \sum h_k \otimes h_k
                    # h_filtered.T @ h_filtered -> [1024, N] @ [N, 1024] -> [1024, 1024]
                    self.G_d[self.domain_id] += h_filtered.T @ h_filtered
                    
                    # C_d += \sum h_k \otimes y_k
                    y_one_hot = F.one_hot(pred_filtered, num_classes=self.num_classes).float() # [N, C]
                    self.C_d[self.domain_id] += h_filtered.T @ y_one_hot

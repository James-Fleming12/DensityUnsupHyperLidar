from torchhd import functional
from torchhd import embeddings

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from faster_mean_shift.mean_shift_cosine_gpu import estimate_bandwidth_binary, mean_shift_binary

class Model(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
        super(Model, self).__init__()

        self.device = device

        # Record the current number of class hypervectors
        self.num_classes = num_classes      # Used in supervised HD
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        # set the input dimension
        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())
        w_dict = torch.load(modeldir + "/SENet_valid_best",
                            map_location=lambda storage, loc: storage)
        self.net.load_state_dict(w_dict['state_dict'], strict=True)
        self.net.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.net.cuda()

        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            # Generate a random projection matrix
            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim, 
                                          randomness=randomness)
            print("self.value", self.value.weight.shape)  # cifar10: [100, 10000] # num_levels * hd_dim
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # cifar10: [1280, 10000]  #bsz x num_features

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        
        else:  # No encoder, use raw samples
            self.hd_dim = self.input_dim

        # Set classify
        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        # self.classify_weights is the sum of all hypervectors, so its scale
        # accounts the number of samples in this class/cluster
        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)
        # print(self.classify_weights.shape)  # size num_class x HD dim

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        # print("x.shape", x.shape)  # torch.Size([1, 5, 64, 512])

        with torch.cuda.amp.autocast(enabled=True):
            x = self.net(x, True)
        
        # print("x.shape", x.shape)  # torch.Size([1, 128, 64, 512])
        # x = self.flatten(x)
        x = x.permute(0, 2, 3, 1)  # shape: (1, 64, 512, 128)
        x = x.reshape(-1, 128)     # shape: (1*64*512, 128) = (32768, 128)
        # sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device)
        # print("x.shape", x.shape)  # torch.Size([32768, 128])
        if PERCENTAGE is not None:
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select
            
            if is_wrong is not None:
                # # Pick by the wrong and keep the PERCENTAGE
                wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
                
                if wrong_indices.numel() >= num_samples:
                    # If there are enough wrong samples, randomly select from them
                    selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                    is_wrong[selected_indices] = False # Mark the selected indices as used
                else:
                    # If there are not enough wrong samples, fill the rest with random samples
                    non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                    remaining = num_samples - wrong_indices.numel()
                    fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]
    
                    selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                    is_wrong[selected_indices] = False # Mark the selected indices as used
            else:
                selected_indices = torch.randperm(x.shape[0], device=x.device)[:num_samples]

            selected_indices, _ = selected_indices.sort()  # Optional: sort to preserve order
            # print("selected_indices", selected_indices.shape)  # e.g., torch.Size([1638])
            x = x[selected_indices]  # shape: (~PERCENTAGE * 32768, 128)
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"

            # Pick by loss: 
            # num_samples = int(x.shape[0] * PERCENTAGE)
            # num_wrongdata = 0
            # sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            # top_indices = sorted_indices[:num_wrongdata]

            # all_indices = torch.arange(is_wrong.shape[0], device=x.device)
            # temp = torch.ones_like(is_wrong, dtype=torch.bool)
            # temp[top_indices] = False
            # remaining_indices = all_indices[temp]

            # remaining = num_samples - num_wrongdata
            # if remaining_indices.numel() >= remaining:
            #     random_fill_indices = remaining_indices[torch.randperm(remaining_indices.shape[0])[:remaining]]
            # else:
            #     # If not enough remaining, take all of them
            #     random_fill_indices = remaining_indices
            
            # selected_indices = torch.cat([top_indices, random_fill_indices], dim=0)
            # is_wrong[selected_indices] = 0 # Mark the selected indices as used

            # Get top losses and their indices (descending sort)
            # sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            # selected_indices = sorted_indices[:num_samples]  # pick top N
            # is_wrong[selected_indices] = 0.0

            # Filter your data
            # x = x[selected_indices]
            # print("x after selection", x.shape)  # e.g., torch.Size([1638, 128])
            # print("x", x[0])  # e.g., torch.Size([1638])

        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            # print("Encode bind value: ", self.value(x)[:, :, mask].shape)  # btz*size x num_features * hd_dim
            # print("Encode position value: ", self.position.weight[:, mask].shape)  # num_features * hd_dim
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])  # bsz*size x num_features x hd_dim
            sample_hv[:, mask] = functional.multiset(tmp_hv)  # bsz*size x hd_dim

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:  # None encoder, just use the raw sample
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        # print("sample_hv.shape", sample_hv.shape)  # (bsz*size, 1000)
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        # Get logits output
        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        # Compute the cosine distance between normalized hypervectors
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))

        #logits = torch.div(logits, self.temperature)
        #softmax_logits = F.log_softmax(logits, dim=1)

        return logits, F.normalize(enc), indices, is_wrong_left # enc is still hd_dim, but some elements are 0

    def get_predictions(self, enc):
        # Compute the cosine distance between normalized hypervectors
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))
        return logits

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:  # self.method == 'BasicHD'
            #class_hv = self.classify_weights / self.classify_sample_cnt
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()

def set_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
    return Model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

class DensityModel(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, max_subclusters = 10, subcluster_type="bipolar", gauss_rp=True, use_adaptor=True):
        super(DensityModel, self).__init__()

        self.device = device
        self.use_adaptor = use_adaptor

        self.num_classes = num_classes
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"], use_adaptor=self.use_adaptor)

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())
            
            if self.ARCH["train"]["pipeline"] == "pointpillar":
                from modules.HDC_cl import PointPillarEncoder

                class _PointPillarEncoder4D(PointPillarEncoder):
                    def forward(self, batch, only_feat=False):
                        return super().forward(batch).unsqueeze(-1).unsqueeze(-1)

                self.net = _PointPillarEncoder4D(
                    in_channels=self.ARCH["train"].get("pointpillar_in_channels", 4),
                    bev_shape=tuple(self.ARCH["train"].get("pointpillar_bev_shape", [512, 512])),
                )

        if self.ARCH["train"]["pipeline"] != "pointpillar":
            w_dict = torch.load(modeldir + "/SENet_valid_best",
                                map_location=lambda storage, loc: storage)
            self.net.load_state_dict(w_dict['state_dict'], strict=True)
            self.net.eval()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.gpu = True
                self.net.cuda()
        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            torch.manual_seed(42) # setting fixed seed for projection initialization (removes saved model randomness)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            if not gauss_rp:
                # self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim) 
                    self.projection.weight.copy_(gaussian_matrix / np.sqrt(self.input_dim))
            else:
                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim)
                    q, _ = torch.linalg.qr(gaussian_matrix)
                    self.projection.weight.copy_(q * torch.sqrt(torch.tensor(self.hd_dim))) # Scale by the square root of the dimension to preserve variance (Johnson-Lindenstrauss)

            torch.set_rng_state(torch_rng_state) # set back to random
            np.random.set_state(numpy_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim, 
                                          randomness=randomness)
            print("self.value", self.value.weight.shape)  # cifar10: [100, 10000] # num_levels * hd_dim
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # cifar10: [1280, 10000]  #bsz x num_features

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)

        self.num_subclusters = max_subclusters
        self.subcluster_type = subcluster_type
        self.subclusters = nn.Parameter(torch.zeros(self.num_classes * self.num_subclusters, self.hd_dim, device=self.device))
        self.subclusters.data.fill_(0.0)

        self.subcluster_to_class = torch.repeat_interleave(torch.arange(self.num_classes, device=self.device), self.num_subclusters)

        self.quantile = 0.4
        self.mult = 0.2
        self.dedup = 0.7

        self.gauss_rp = gauss_rp

        self.register_buffer('proto_momentum', torch.zeros_like(self.classify.weight.data)) # EMA momentum

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None, chunk_idx=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.amp.autocast('cuda', enabled=True):
            x = self.net(x, only_feat=True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        if chunk_idx is not None:
            start, end = chunk_idx
            x = x[start:end]

        if PERCENTAGE is not None:
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select

            if wrong_indices.numel() >= num_samples:
                selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                is_wrong[selected_indices] = False
            else:
                non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                remaining = num_samples - wrong_indices.numel()
                fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]

                selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                is_wrong[selected_indices] = False

            selected_indices, _ = selected_indices.sort()
            x = x[selected_indices]
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"
        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])
            sample_hv[:, mask] = functional.multiset(tmp_hv)

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))

        return logits, F.normalize(enc), indices, is_wrong_left

    def get_predictions(self, enc):
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))
        return logits

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()
    
    def update_subclusters(self, x, proj_labels, learning_rate=0.1, min_samples=10, method="proximity_pull"):
        """
        Updates subclusters from labeled data.
        method: 'proximity_pull', 'soft_weighted', 'mean_shift'
        """
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            labels_flat = proj_labels.view(-1)

            assert enc.shape[0] == labels_flat.shape[0], f"Encoding size {enc.shape[0]} doesn't match label size {labels_flat.shape[0]}"

            for class_id in range(self.num_classes):
                class_mask = labels_flat == class_id
                if class_mask.sum() < min_samples:
                    continue

                class_enc = enc[class_mask].float()
                mask = self.subcluster_to_class == class_id
                relevant_subclusters = self.subclusters[mask]
                subcluster_indices = torch.nonzero(mask).squeeze(1)
                n_subs = relevant_subclusters.shape[0]

                if self.subcluster_type == 'bipolar':
                    class_enc_binary = torch.sign(class_enc).to(dtype=self.subclusters.dtype)
                    similarities = (torch.matmul(class_enc_binary, relevant_subclusters.T) + self.hd_dim) / (2 * self.hd_dim)
                else:
                    similarities = torch.matmul(F.normalize(class_enc, dim=1), F.normalize(relevant_subclusters, dim=1).T)

                if method == "proximity_pull": # each sample updates only its closest subcluster
                    assignments = torch.argmax(similarities, dim=1)
                    assignments_expanded = assignments.unsqueeze(1).expand(-1, class_enc.shape[1])

                    sum_per_sub = torch.zeros(n_subs, class_enc.shape[1], device=self.device, dtype=torch.float32)
                    sum_per_sub.scatter_add_(0, assignments_expanded, class_enc)

                    counts = torch.zeros(n_subs, device=self.device, dtype=torch.float32)
                    counts.scatter_add_(0, assignments, torch.ones(assignments.shape[0], device=self.device))

                    valid = counts >= min_samples
                    if not valid.any():
                        continue

                    new_means = sum_per_sub[valid] / counts[valid].unsqueeze(1)
                elif method == "soft_weighted": # each sample contributes to all subclusters weighted by similarity
                    weights = F.softmax(similarities, dim=1)

                    new_means = torch.matmul(weights.T, class_enc)
                    weight_sums = weights.sum(dim=0)

                    valid = weight_sums >= (min_samples * 0.1)
                    if not valid.any():
                        continue

                    new_means = new_means[valid] / weight_sums[valid].unsqueeze(1)
                elif method == "mean_shift":
                    class_emb_np = class_enc.cpu().numpy()

                    estimated_bandwidth = estimate_bandwidth_binary(
                        class_emb_np,
                        quantile=self.quantile,
                        n_samples=min(500, len(class_emb_np)),
                        bandwidth_multiplier=self.mult
                    )

                    try:
                        new_centers = mean_shift_binary(
                            X=class_emb_np,
                            bandwidth=estimated_bandwidth,
                            quantile=self.quantile,
                            bandwidth_multiplier=self.mult,
                            dedup_scale=self.dedup
                        )
                    except Exception as e:
                        print(f"  Mean shift failed for class {class_id}: {e}, skipping")
                        continue

                    new_centers = np.sign(new_centers)
                    new_centers_t = F.normalize(torch.tensor(new_centers, dtype=torch.float32, device=self.device), dim=1)

                    for new_center in new_centers_t: # find similar existing subcluster and pull towards
                        if self.subcluster_type == 'bipolar':
                            new_binary = torch.sign(new_center).to(dtype=self.subclusters.dtype)
                            sims = (torch.matmul(new_binary.unsqueeze(0), relevant_subclusters.T) + self.hd_dim) / (2 * self.hd_dim)
                        else:
                            sims = torch.matmul(F.normalize(new_center.unsqueeze(0), dim=1), F.normalize(relevant_subclusters, dim=1).T)

                        sims = sims.squeeze(0)
                        closest_idx = sims.argmax().item()
                        absolute_idx = subcluster_indices[closest_idx].item()

                        current = self.subclusters.data[absolute_idx].float()
                        updated = (1.0 - learning_rate) * current + learning_rate * new_center.float()

                        if self.subcluster_type == 'bipolar':
                            updated = torch.sign(updated)
                            updated[updated == 0] = -1.0

                        self.subclusters.data[absolute_idx] = F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)

                        relevant_subclusters = self.subclusters[mask]

                    continue # skip the rest of the update logic (only used with the other two methods)
                else:
                    raise ValueError(f"Unknown method: {method}. Choose 'proximity_pull' or 'soft_weighted'.")

                if self.subcluster_type == 'bipolar': # Normalize new means
                    new_means = torch.sign(new_means)
                    new_means[new_means == 0] = -1.0

                new_means = F.normalize(new_means, dim=1)

                valid_absolute_indices = subcluster_indices[valid]
                current = self.subclusters.data[valid_absolute_indices].float()
                updated = (1.0 - learning_rate) * current + learning_rate * new_means # EMA Update

                if self.subcluster_type == 'bipolar':
                    updated = torch.sign(updated)
                    updated[updated == 0] = -1.0

                self.subclusters.data[valid_absolute_indices] = F.normalize(updated, dim=1)

    def init_subclusters(self, dataloader, bandwidth=None, max_samples_per_class=8000, sampling_strategy='diverse'):
        """
        sampling_strategy: 'random' (simple random sampling), 'diverse' (stratified, temporal diversity), or 'fps' (farthest point sampling)
        """
        self.eval()
        num_sub_per_cluster = self.num_subclusters
        print(f"Collecting embeddings for {self.num_classes} classes using '{sampling_strategy}' sampling")
        all_subcluster_centers = []
        all_subcluster_classes = []

        MAX_SAMPLES = max_samples_per_class * 2
        
        for class_id in range(self.num_classes):
            print(f"Processing class {class_id}...")
            class_embeddings = []
            batch_indices = []
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(dataloader):
                    
                    if isinstance(proj_in, dict): 
                        # for AI Motive
                        proj_in = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in proj_in.items()}
                    else:
                        proj_in = proj_in.to(self.device)
                    
                    proj_labels = proj_labels.to(self.device).flatten()

                    valid_label_mask = proj_labels >= 0
                    if not valid_label_mask.any():
                        del proj_in, proj_labels
                        self._clear_memory()
                        continue
                    proj_labels = proj_labels[valid_label_mask]
                    enc, _, _ = self.encode(proj_in)
                    enc = enc[valid_label_mask]

                    class_mask = proj_labels == class_id
                    
                    if torch.any(class_mask):
                        class_enc = enc[class_mask].cpu().half()
                        class_embeddings.append(class_enc)
                        batch_indices.extend([batch_idx] * class_enc.shape[0])
                        total_samples += class_enc.shape[0]
                    
                    del proj_in, proj_labels
                    self._clear_memory()

                    if total_samples >= MAX_SAMPLES: # collect extra for better sampling
                        break
            
            if not class_embeddings:
                print(f"  No data for class {class_id}, skipping")
                continue
            
            class_emb_cpu = torch.cat(class_embeddings, dim=0)

            if len(class_emb_cpu) > MAX_SAMPLES:
                indices = torch.randperm(len(class_emb_cpu))[:MAX_SAMPLES]
                class_emb_cpu = class_emb_cpu[indices]
            
            batch_indices = torch.as_tensor(batch_indices[:len(class_emb_cpu)])
            batch_indices = batch_indices.detach().clone()

            if len(class_emb_cpu) > max_samples_per_class:
                if sampling_strategy == 'random':
                    indices = torch.randperm(len(class_emb_cpu))[:max_samples_per_class]
                elif sampling_strategy == 'diverse':
                    indices = self._stratified_sample(batch_indices, max_samples_per_class)
                elif sampling_strategy == 'fps':
                    indices = self._farthest_point_sample(class_emb_cpu, max_samples_per_class)
                else:
                    raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
                
                class_emb_cpu = class_emb_cpu[indices]
                print(f"  Sampled {len(class_emb_cpu)} from {len(batch_indices)} total samples using '{sampling_strategy}'")
            
            class_emb_np = class_emb_cpu.numpy()
            
            if bandwidth is None:
                estimated_bandwidth = estimate_bandwidth_binary(
                    class_emb_np, 
                    quantile=self.quantile,
                    n_samples=min(500, len(class_emb_np)), # just making it quicker (hopefully not an issue)
                    bandwidth_multiplier=self.mult
                )
                print(f"  Estimated bandwidth for class {class_id}: {estimated_bandwidth:.4f}")
                class_bandwidth = estimated_bandwidth
            else:
                class_bandwidth = bandwidth
            
            print(f"  Using {len(class_emb_np)} samples for clustering")
            
            del class_emb_cpu, class_embeddings
            self._clear_memory()

            subclusters_for_class = self._process_single_class(
                class_emb_np, class_id, num_sub_per_cluster, class_bandwidth
            )
            
            all_subcluster_centers.extend(subclusters_for_class)
            all_subcluster_classes.extend([class_id] * len(subclusters_for_class))
            
            del class_emb_np
            self._clear_memory()
        
        self._load_subclusters(all_subcluster_centers, all_subcluster_classes)
        print("Subcluster initialization complete")

    def _stratified_sample(self, batch_indices, n_samples):
        unique_batches = torch.unique(batch_indices)
        samples_per_batch = n_samples // len(unique_batches)
        remainder = n_samples % len(unique_batches)
        
        selected_indices = []
        for i, batch_id in enumerate(unique_batches):
            batch_mask = batch_indices == batch_id
            batch_positions = torch.where(batch_mask)[0]

            n_from_batch = samples_per_batch + (1 if i < remainder else 0)
            n_from_batch = min(n_from_batch, len(batch_positions))

            perm = torch.randperm(len(batch_positions))[:n_from_batch]
            selected_indices.append(batch_positions[perm])
        
        return torch.cat(selected_indices)

    def _farthest_point_sample(self, embeddings, n_samples):
        n_points = len(embeddings)
        if n_points <= n_samples:
            return torch.arange(n_points)
        
        selected = [torch.randint(0, n_points, (1,)).item()]
        distances = torch.full((n_points,), float('inf'))
        
        for _ in range(n_samples - 1):
            last_selected = embeddings[selected[-1]]
            new_distances = torch.sum((embeddings - last_selected) ** 2, dim=1)

            distances = torch.minimum(distances, new_distances)

            farthest_idx = torch.argmax(distances).item()
            selected.append(farthest_idx)

            distances[farthest_idx] = 0
        
        return torch.tensor(selected)

    def _process_single_class(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
        """Process a single class to generate its subclusters using Adaptive KNN bandwidth estimation."""
        if len(class_emb_np) == 0:
            return []
        
        print(f"  Running Adaptive KNN clustering on {len(class_emb_np)} samples...")
        K = 15
        H = torch.tensor(class_emb_np, device=self.device, dtype=torch.float32)

        if len(H) > 2000:
            H_sample = H[torch.randperm(len(H))[:2000]]
        else:
            H_sample = H
            
        H_sample = torch.nn.functional.normalize(H_sample, dim=1)
        sim_matrix = torch.matmul(H_sample, H_sample.T)

        topk_sims, _ = torch.topk(sim_matrix, min(K+1, len(H_sample)), dim=1)
        if topk_sims.shape[1] > 1:
            topk_dists = 1.0 - topk_sims[:, 1:] 
            sigma_i = topk_dists.mean(dim=1)
            sigma_i = torch.clamp(sigma_i, min=1e-3)
        else:
            sigma_i = torch.ones(len(H_sample), device=self.device) * 0.1
        
        centers = H_sample.clone()
        for _ in range(15):
            sim = torch.matmul(H_sample, centers.T)
            dist = 1.0 - sim

            sigma = sigma_i.unsqueeze(1)
            weights = torch.exp(-(dist ** 2) / (2 * (sigma ** 2) + 1e-8))

            new_centers = torch.matmul(weights.T, H_sample)
            new_centers = torch.nn.functional.normalize(new_centers, dim=1)
            
            shift = torch.norm(new_centers - centers, dim=1).max()
            centers = new_centers
            if shift < 1e-4: break

        center_sim = torch.matmul(centers, centers.T)
        keep = torch.ones(len(centers), dtype=torch.bool, device=self.device)
        for i in range(len(centers)):
            if keep[i]:
                close = center_sim[i] > 0.95
                close[i] = False
                keep[close] = False
                
        unique_centers = centers[keep]
        
        if self.subcluster_type == "bipolar":
            unique_centers = torch.sign(unique_centers)
            unique_centers[unique_centers == 0] = -1.0 # Standardize to -1, 1

        num_clusters_found = len(unique_centers)
        print(f"  Found {num_clusters_found} clusters")

        subclusters = []
        if num_clusters_found <= num_sub_per_cluster:
            for center in unique_centers:
                subclusters.append(center.cpu())
        else:
            unique_centers_cpu = unique_centers.cpu()
            fps_indices = self._farthest_point_sample(unique_centers_cpu, num_sub_per_cluster)
            for idx in fps_indices.tolist():
                subclusters.append(unique_centers_cpu[idx])

        return subclusters

    def _load_subclusters(self, centers_list, classes_list):
        """Load subclusters into model parameters with memory efficiency."""
        if not centers_list:
            print("Warning: No subclusters to load")
            return
        
        total_centers = len(centers_list)
        print(f"Loading {total_centers} subclusters into model (type: {self.subcluster_type})...")

        with torch.no_grad():
            batch_size = 100
            for i in range(0, total_centers, batch_size):
                end_idx = min(i + batch_size, total_centers)

                if self.subcluster_type == 'bipolar':
                    batch = torch.stack([self._make_bipolar(c.to(self.device)) for c in centers_list[i:end_idx]])
                elif self.subcluster_type == 'continuous':
                    batch = torch.stack([c.to(self.device) if c.device.type == 'cpu' else c for c in centers_list[i:end_idx]])
                    batch = F.normalize(batch, dim=1)

                    norms = torch.norm(batch, dim=1)
                    expected_norms = torch.ones(len(batch), device=self.device, dtype=batch.dtype)
                    assert torch.allclose(norms, expected_norms, atol=1e-3), f"Continuous subclusters must be unit norm! Got norms: {norms}"                
                else:
                    raise ValueError(f"Unknown subcluster_type: {self.subcluster_type}")

                self.subclusters.data[i:end_idx] = batch

                del batch
                if i % 500 == 0:
                    self._clear_memory()
                    print(f"  Loaded {end_idx}/{total_centers} subclusters")

        print("All subclusters loaded")

    def _make_bipolar(self, tensor):
        """Convert tensor to bipolar {-1, +1}, mapping 0 -> 1."""
        # Method 1: Map zeros to +1 (most common)
        result = torch.sign(tensor)
        result[result == 0] = -1
        return result

    def _clear_memory(self):
        """Aggressive memory clearing."""
        # import gc
        # gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def get_max_subcluster_similarity(self, enc, class_id, distance_sensitivity=1.0):
        """
        Get maximum similarity [0,1] to subclusters.
        Handles both bipolar and continuous subclusters.
        """
        mask = self.subcluster_to_class == class_id
        relevant_subclusters = self.subclusters[mask].float()

        if self.subcluster_type == 'bipolar':
            enc_binary = torch.sign(enc).float()
            hd_dim = enc_binary.shape[1]
            
            dot_products = torch.matmul(enc_binary, relevant_subclusters.T)

            base_similarity = (dot_products + hd_dim) / (2 * hd_dim)
        elif self.subcluster_type == 'continuous':
            enc_norm = F.normalize(enc.float(), dim=1)
            sub_norm = F.normalize(relevant_subclusters, dim=1)
            cosine_sim = torch.matmul(enc_norm, sub_norm.T)

            base_similarity = (cosine_sim + 1) / 2
        else:
            raise ValueError(f"Unknown subcluster_type: {self.subcluster_type}")

        if distance_sensitivity == 0.0:
            scaled_similarity = torch.where(base_similarity > 0.5, torch.tensor(1.0, device=enc.device), base_similarity * 2.0)
        elif distance_sensitivity == 1.0:
            scaled_similarity = base_similarity
        else:
            scaled_similarity = base_similarity ** distance_sensitivity

        max_similarities, relative_indices = torch.max(scaled_similarity, dim=1)
        absolute_indices = torch.nonzero(mask)[relative_indices, 0]
        
        return max_similarities, absolute_indices

    def inference_update(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80]):
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]

            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1) # ignore background from updates
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            num_active = active_enc.shape[0]

            curr_chunk_size = num_active if chunk_size == -1 else chunk_size # handle chunk_size=-1 to remove chunking altogether

            all_predictions = []
            all_update_masks = []

            if self.subcluster_type == 'bipolar':
                proto_binary = torch.sign(self.classify.weight)

            for i in range(0, num_active, curr_chunk_size):
                chunk_enc_norm = enc_norm[i : i + curr_chunk_size]
                chunk_logits = self.classify(chunk_enc_norm)
                chunk_preds = torch.argmax(chunk_logits, dim=1)
                all_predictions.append(chunk_preds)

                if self.subcluster_type == 'bipolar':
                    chunk_enc_orig = active_enc[i : i + curr_chunk_size]
                    enc_binary = torch.sign(chunk_enc_orig)
                    selected_proto = proto_binary[chunk_preds]
                    sims = torch.sum(enc_binary * selected_proto, dim=1) / self.hd_dim
                else:
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
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                # proto_sims = torch.sum(sample_encs * F.normalize(self.classify.weight[c_id].unsqueeze(0), dim=1), dim=1)
                # proto_valid = proto_sims < thresholds[1] # masks based on proximity to class prototype
                # if not torch.any(proto_valid):
                #     continue
                # class_indices = class_indices[proto_valid]
                # sample_encs = sample_encs[proto_valid]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0] # masks based on proximity to class subclusters
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()  # normalize weights to sum to 1
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_ema(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], alpha=0.999):
        """Orthogonalized Exponential Moving Average Update"""
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
            curr_chunk_size = num_active if chunk_size == -1 else chunk_size

            all_predictions = []
            all_update_masks = []

            if self.subcluster_type == 'bipolar':
                proto_binary = torch.sign(self.classify.weight)

            for i in range(0, num_active, curr_chunk_size):
                chunk_enc_norm = enc_norm[i : i + curr_chunk_size]
                chunk_logits = self.classify(chunk_enc_norm)
                chunk_preds = torch.argmax(chunk_logits, dim=1)
                all_predictions.append(chunk_preds)

                if self.subcluster_type == 'bipolar':
                    chunk_enc_orig = active_enc[i : i + curr_chunk_size]
                    enc_binary = torch.sign(chunk_enc_orig)
                    selected_proto = proto_binary[chunk_preds]
                    sims = torch.sum(enc_binary * selected_proto, dim=1) / self.hd_dim
                else:
                    selected_proto = F.normalize(self.classify.weight[chunk_preds])
                    sims = torch.sum(chunk_enc_norm * selected_proto, dim=1)

    def inference_update_srp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80]):
        """Subcluster-Regularized Pull Update"""
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
                
            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            
            normalized_prototypes = F.normalize(self.classify.weight)
            sims = F.linear(enc_norm, normalized_prototypes)
            max_sims, _ = sims.max(dim=1)
            update_mask = max_sims > thresholds[0]
            
            for class_id in unique_classes:
                c_id = class_id.item()

                class_mask = (predictions == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                
                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                weights_sample = sub_sims / sub_sims.sum()
                sample_pull_vector = (sample_encs * weights_sample.unsqueeze(1)).sum(dim=0).float()
                
                # Subcluster regularization
                matched_subclusters = self.subclusters.data[sub_indices].float()
                subcluster_pull_vector = (matched_subclusters * weights_sample.unsqueeze(1)).sum(dim=0).float()
                
                # 80/20 mix
                weighted_pull_vector = 0.8 * sample_pull_vector + 0.2 * subcluster_pull_vector
                
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_awd(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80]):
        """Activity-Weighted Distillation Update"""
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
                
            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            
            normalized_prototypes = F.normalize(self.classify.weight)
            sims = F.linear(enc_norm, normalized_prototypes)
            max_sims, _ = sims.max(dim=1)
            update_mask = max_sims > thresholds[0]
            
            for class_id in unique_classes:
                c_id = class_id.item()

                class_mask = (predictions == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                sample_encs = enc_norm[class_indices]
                
                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, max_sub_indices = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, max_sub_indices = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                max_sub_indices = max_sub_indices[valid_mask]

                unique_subs, sub_counts = torch.unique(max_sub_indices, return_counts=True)
                total_hits = sub_counts.sum().float()
                
                for sub_idx in unique_subs:
                    s_mask = max_sub_indices == sub_idx
                    s_samples = sample_encs[s_mask]
                    s_sims = sub_sims[s_mask]
                    
                    weights_sample = s_sims / s_sims.sum()
                    pull_vec = (s_samples * weights_sample.unsqueeze(1)).sum(dim=0).float()
                    
                    current_sub = self.subclusters.data[sub_idx].float()
                    
                    subcluster_lr = learning_rate * s_sims.mean().item()
                    updated_sub = (1.0 - subcluster_lr) * current_sub + subcluster_lr * pull_vec
                    
                    if self.subcluster_type == 'continuous':
                        updated_sub = F.normalize(updated_sub.unsqueeze(0), dim=1).squeeze(0)
                    else:
                        updated_sub = torch.sign(updated_sub)
                        
                    self.subclusters.data[sub_idx] = updated_sub.to(self.subclusters.dtype)

                # Distill using Activity Weighting
                active_subclusters = self.subclusters.data[unique_subs].float()
                
                weights = (sub_counts.float() / total_hits).unsqueeze(1)
                distilled_prototype = (active_subclusters * weights).sum(dim=0)
                
                effective_lr = learning_rate * sub_sims.mean().item()
                current_weight = self.classify.weight[c_id].float()
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * distilled_prototype
                
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_cwd(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], temp=0.1):
        """Confidence-Weighted Distillation Update"""
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
                
            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            
            normalized_prototypes = F.normalize(self.classify.weight)
            sims = F.linear(enc_norm, normalized_prototypes)
            max_sims, _ = sims.max(dim=1)
            update_mask = max_sims > thresholds[0]
            
            for class_id in unique_classes:
                c_id = class_id.item()

                class_mask = (predictions == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                sample_encs = enc_norm[class_indices]
                
                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, max_sub_indices = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, max_sub_indices = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                max_sub_indices = max_sub_indices[valid_mask]

                unique_subs = torch.unique(max_sub_indices)
                for sub_idx in unique_subs:
                    s_mask = max_sub_indices == sub_idx
                    s_samples = sample_encs[s_mask]
                    s_sims = sub_sims[s_mask]
                    
                    weights_sample = s_sims / s_sims.sum()
                    pull_vec = (s_samples * weights_sample.unsqueeze(1)).sum(dim=0).float()
                    
                    current_sub = self.subclusters.data[sub_idx].float()
                    
                    subcluster_lr = learning_rate * s_sims.mean().item()
                    updated_sub = (1.0 - subcluster_lr) * current_sub + subcluster_lr * pull_vec
                    
                    if self.subcluster_type == 'continuous':
                        updated_sub = F.normalize(updated_sub.unsqueeze(0), dim=1).squeeze(0)
                    else:
                        updated_sub = torch.sign(updated_sub)
                        
                    self.subclusters.data[sub_idx] = updated_sub.to(self.subclusters.dtype)

                # Confidence-Weighted Distillation
                start_idx = c_id * self.num_subclusters
                end_idx = start_idx + self.num_subclusters
                class_subclusters = self.subclusters.data[start_idx:end_idx].float()
                
                current_weight = self.classify.weight[c_id].float()
                sub_sims = F.linear(class_subclusters, current_weight.unsqueeze(0)).squeeze()
                
                # Softmax over similarities to ignore noisy subclusters
                distill_weights = F.softmax(sub_sims / temp, dim=0).unsqueeze(1)
                distilled_prototype = (class_subclusters * distill_weights).sum(dim=0)
                
                effective_lr = learning_rate * sub_sims.mean().item()
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * distilled_prototype
                
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_psp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80]):
        """Prototype-Subcluster Ping-Pong Update"""
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
                
            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            
            normalized_prototypes = F.normalize(self.classify.weight)
            sims = F.linear(enc_norm, normalized_prototypes)
            max_sims, _ = sims.max(dim=1)
            update_mask = max_sims > thresholds[0]
            
            for class_id in unique_classes:
                c_id = class_id.item()

                class_mask = (predictions == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                
                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                weights_sample = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights_sample.unsqueeze(1)).sum(dim=0).float()
                
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                updated_weight = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                self.classify.weight[c_id] = updated_weight.to(self.classify.weight.dtype)
                
                # Ping-Pong: Gently pull all subclusters for this class towards the NEW master prototype
                start_idx = c_id * self.num_subclusters
                end_idx = start_idx + self.num_subclusters
                
                # Use a much smaller learning rate to preserve their spread
                sub_pull_lr = effective_lr * 0.1
                current_subs = self.subclusters.data[start_idx:end_idx].float()
                updated_subs = (1.0 - sub_pull_lr) * current_subs + sub_pull_lr * updated_weight.unsqueeze(0)
                
                if self.subcluster_type == 'continuous':
                    updated_subs = F.normalize(updated_subs, dim=1)
                else:
                    updated_subs = torch.sign(updated_subs)
                    
                self.subclusters.data[start_idx:end_idx] = updated_subs.to(self.subclusters.dtype)

            return full_predictions
    def inference_update_tcg(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Temporal Consistency Gating"""
        if not hasattr(self, 'tcg_buffer'):
            self.tcg_buffer = []

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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            self.tcg_buffer.append(full_predictions)
            if len(self.tcg_buffer) > 3:
                self.tcg_buffer.pop(0)
            
            if len(self.tcg_buffer) < 3:
                return full_predictions

            temporal_mask = (self.tcg_buffer[0] == self.tcg_buffer[1]) & (self.tcg_buffer[1] == self.tcg_buffer[2])
            temporal_mask = temporal_mask[valid_enc_mask]

            unique_classes = torch.unique(predictions[temporal_mask])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id) & temporal_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0).float()
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_ogaa(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Oracle-Guided Active Anchoring"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            if oracle_labels is not None:
                labels_flat = oracle_labels.view(-1)
                active_labels = labels_flat[valid_enc_mask]

                top2_logits, _ = torch.topk(chunk_logits, 2, dim=1)
                margin = top2_logits[:, 0] - top2_logits[:, 1]

                num_to_label = min(5, active_enc.shape[0])
                _, confusing_indices = torch.topk(margin, num_to_label, largest=False)

                for idx in confusing_indices:
                    gt_label = active_labels[idx].item()
                    if gt_label <= 0 or gt_label == 255: 
                        continue
                    
                    sample_enc = enc_norm[idx].float()
                    massive_lr = 0.5 
                    
                    current_weight = self.classify.weight[gt_label].float()
                    updated_weight = (1.0 - massive_lr) * current_weight + massive_lr * sample_enc
                    self.classify.weight[gt_label] = F.normalize(updated_weight.unsqueeze(0), dim=0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_srt(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Subcluster-Routed Translation"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                sub_indices = sub_indices[valid_mask]

                matched_subclusters = self.subclusters.data[sub_indices].float()
                
                translation_vectors = sample_encs.float() - matched_subclusters
                
                weights = sub_sims / sub_sims.sum()
                weighted_translation_vector = (translation_vectors * weights.unsqueeze(1)).sum(dim=0)
                
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_translation_vector).to(self.proto_momentum.dtype)
                
                updated_weight = current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dmrp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Decoupled Memory-Replay Pull"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0).float()
                
                mask_sub = self.subcluster_to_class == c_id
                relevant_subclusters = self.subclusters[mask_sub].float()
                random_idx = torch.randint(0, relevant_subclusters.shape[0], (1,)).item()
                replay_vector = relevant_subclusters[random_idx]

                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr * 2) * current_weight + effective_lr * self.proto_momentum[c_id].float() + effective_lr * replay_vector
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_ovsp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Oracle-Verified Soft Pull"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            
            if oracle_labels is not None:
                labels_flat = oracle_labels.view(-1)[valid_enc_mask]
                top2_logits, _ = torch.topk(chunk_logits, 2, dim=1)
                margin = top2_logits[:, 0] - top2_logits[:, 1]
                
                num_to_label = min(5, active_enc.shape[0])
                _, confusing_indices = torch.topk(margin, num_to_label, largest=False)
                
                synthetic_conf = torch.zeros(active_enc.shape[0], device=self.device, dtype=torch.float32)
                for idx in confusing_indices:
                    gt_label = labels_flat[idx].item()
                    if gt_label > 0 and gt_label != 255:
                        predictions[idx] = gt_label
                        synthetic_conf[idx] = 1.0

            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                if oracle_labels is not None:
                    syn_c = synthetic_conf[class_indices]
                    override_mask = syn_c > 0
                    sub_sims[override_mask] = syn_c[override_mask]

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dcsp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Density-Calibrated Standard Pull"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            active_ranges = x_flat[valid_enc_mask, 0] 
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                
                sample_ranges = active_ranges[class_indices][valid_mask].abs()
                range_scale = sample_ranges / (sample_ranges.max() + 1e-4)
                
                combined_weight = sub_sims * range_scale
                weights = combined_weight / combined_weight.sum()

                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_cacg(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Cross-Augmentation Consistency Gating"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            
            x_aug = torch.roll(x, shifts=2, dims=3)
            enc_aug, _, _ = self.encode(x_aug)
            
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            active_enc_aug = enc_aug[valid_enc_mask]
            
            enc_norm = F.normalize(active_enc)
            enc_norm_aug = F.normalize(active_enc_aug)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                enc_norm_aug = enc_norm_aug.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            
            chunk_logits_aug = self.classify(enc_norm_aug)
            predictions_aug = chunk_logits_aug.argmax(dim=1)
            
            consistency_mask = (predictions == predictions_aug)

            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id) & consistency_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dbmr(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Dual-Buffer Memory Replay"""
        if getattr(self, 'dbmr_target_buffer', None) is None:
            self.dbmr_target_buffer = {c: [] for c in range(self.num_classes)}
            self.dbmr_source_buffer = {c: [] for c in range(self.num_classes)}
            for i in range(len(self.subclusters.data)):
                c = self.subcluster_to_class[i].item()
                self.dbmr_source_buffer[c].append(self.subclusters.data[i].clone())
        
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0).float()
                
                high_conf_mask = sub_sims > thresholds[1]
                if torch.any(high_conf_mask):
                    hc_samples = sample_encs[high_conf_mask]
                    for hc_s in hc_samples:
                        self.dbmr_target_buffer[c_id].append(hc_s.clone())
                        if len(self.dbmr_target_buffer[c_id]) > 50:
                            self.dbmr_target_buffer[c_id].pop(0)

                effective_lr = learning_rate * sub_sims.mean().item()

                source_pull = torch.zeros_like(weighted_pull_vector)
                target_pull = torch.zeros_like(weighted_pull_vector)
                
                if len(self.dbmr_source_buffer[c_id]) > 0:
                    idx = torch.randint(0, len(self.dbmr_source_buffer[c_id]), (1,)).item()
                    source_pull = self.dbmr_source_buffer[c_id][idx].float()
                
                if len(self.dbmr_target_buffer[c_id]) > 0:
                    idx = torch.randint(0, len(self.dbmr_target_buffer[c_id]), (1,)).item()
                    target_pull = self.dbmr_target_buffer[c_id][idx].float()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr * 3) * current_weight + \
                                 effective_lr * self.proto_momentum[c_id].float() + \
                                 effective_lr * source_pull + \
                                 effective_lr * target_pull

                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dcsp_fix(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Class-Normalized Density Clamping"""
        if getattr(self, 'expected_class_density', None) is None:
            self.expected_class_density = torch.zeros(self.num_classes, device=self.device)
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            active_ranges = x_flat[valid_enc_mask, 0].abs()
            active_densities = 1.0 / (active_ranges + 1e-4)
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                
                class_densities = active_densities[class_indices][valid_mask]
                mean_density = class_densities.mean()
                if self.expected_class_density[c_id] == 0:
                    self.expected_class_density[c_id] = mean_density
                else:
                    self.expected_class_density[c_id] = 0.99 * self.expected_class_density[c_id] + 0.01 * mean_density

                ratio = class_densities / (self.expected_class_density[c_id] + 1e-6)
                ratio = torch.clamp(ratio, max=1.5)
                
                combined_weight = sub_sims * ratio
                weights = combined_weight / combined_weight.sum()

                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_mjcg(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Multi-Jitter Consensus Gating"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            x_aug1 = torch.roll(x, shifts=1, dims=3)
            x_aug2 = torch.roll(x, shifts=-1, dims=3)
            enc1, _, _ = self.encode(x_aug1)
            enc2, _, _ = self.encode(x_aug2)
            
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            active_enc1 = enc1[valid_enc_mask]
            active_enc2 = enc2[valid_enc_mask]
            
            enc_norm = F.normalize(active_enc)
            enc_norm1 = F.normalize(active_enc1)
            enc_norm2 = F.normalize(active_enc2)
            
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                enc_norm1 = enc_norm1.to(self.classify.weight.dtype)
                enc_norm2 = enc_norm2.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            preds = chunk_logits.argmax(dim=1)
            
            preds1 = self.classify(enc_norm1).argmax(dim=1)
            preds2 = self.classify(enc_norm2).argmax(dim=1)
            
            consistency_mask = (preds == preds1) & (preds == preds2)

            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=preds.dtype)
            full_predictions[valid_enc_mask] = preds

            unique_classes = torch.unique(preds)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & consistency_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_knnspp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """K-Nearest Sub-Prototype Pull"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                sub_indices = sub_indices[valid_mask]
                
                unique_subs, inv_idx = torch.unique(sub_indices, return_inverse=True)
                for i, abs_idx in enumerate(unique_subs.tolist()):
                    member_mask = inv_idx == i
                    member_encs = sample_encs[member_mask]
                    member_sims = sub_sims[member_mask]
                    w = member_sims / member_sims.sum()
                    pull_vec = (member_encs * w.unsqueeze(1)).sum(dim=0)
                    
                    current_sub = self.subclusters.data[abs_idx].float()
                    eff_lr = learning_rate * member_sims.mean().item()
                    updated_sub = (1.0 - eff_lr) * current_sub + eff_lr * pull_vec
                    if self.subcluster_type == 'bipolar':
                        updated_sub = torch.sign(updated_sub)
                        updated_sub[updated_sub == 0] = -1.0
                    self.subclusters.data[abs_idx] = F.normalize(updated_sub.unsqueeze(0), dim=0).to(self.subclusters.data.dtype)

                mask_sub = self.subcluster_to_class == c_id
                relevant_subs = self.subclusters.data[mask_sub].float()
                mean_sub = relevant_subs.mean(dim=0)
                self.classify.weight[c_id] = F.normalize(mean_sub.unsqueeze(0), dim=0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_tpda(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Two-Pass Distribution Alignment"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            chunk_dist = torch.bincount(predictions, minlength=self.num_classes).float()
            chunk_dist = chunk_dist / (chunk_dist.sum() + 1e-6)

            if getattr(self, 'tpda_prior', None) is None:
                self.tpda_prior = chunk_dist.clone()
            else:
                self.tpda_prior = 0.99 * self.tpda_prior + 0.01 * chunk_dist

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                
                prior = self.tpda_prior[c_id].item()
                current = chunk_dist[c_id].item()
                penalty_scale = min(1.0, prior / (current + 1e-6))
                
                effective_lr = learning_rate * sub_sims.mean().item() * penalty_scale

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_evuq(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Equal-Volume Update Queues"""
        if getattr(self, 'evuq_queues', None) is None:
            self.evuq_queues = {c: [] for c in range(self.num_classes)}
            
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                valid_encs = sample_encs[valid_mask]
                
                for i in range(valid_encs.shape[0]):
                    self.evuq_queues[c_id].append(valid_encs[i].unsqueeze(0))
                
                while len(self.evuq_queues[c_id]) >= 100:
                    batch = torch.cat(self.evuq_queues[c_id][:100], dim=0)
                    self.evuq_queues[c_id] = self.evuq_queues[c_id][100:]
                    
                    mean_vec = batch.mean(dim=0)
                    
                    effective_lr = learning_rate
                    current_weight = self.classify.weight[c_id].float()
                    self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * mean_vec).to(self.proto_momentum.dtype)
                    
                    updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                    self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dcpm(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Dynamic Class-Paced Momentum"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                M_c = len(sample_encs)
                alpha_c = 1.0 - (learning_rate / (M_c + 1e-6))
                alpha_c = max(0.0, min(1.0, alpha_c))

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = alpha_c * current_weight + (1.0 - alpha_c) * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_pcsg(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Prior-Calibrated Similarity Gating"""
        if getattr(self, 'pcsg_prior', None) is None:
            p = torch.tensor([0.4, 0.2, 0.1, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01], device=self.device)
            p = p[:self.num_classes]
            p = p / p.sum()
            self.pcsg_prior = p
            
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                gamma = 0.05
                p_c = self.pcsg_prior[c_id]
                adjusted_sims = sub_sims - gamma * torch.log(p_c + 1e-6)

                valid_mask = adjusted_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dcdrp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Density-Calibrated Dual-Rate Pull"""
        if getattr(self, 'source_prototypes_dcdrp', None) is None:
            self.source_prototypes_dcdrp = self.classify.weight.clone().detach()

        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            active_ranges = x_flat[valid_enc_mask, 0].abs()
            active_densities = 1.0 / (active_ranges + 1e-4)

            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                
                current_w = self.classify.weight[c_id].float()
                source_w = self.source_prototypes_dcdrp[c_id].float()
                cos_dist = 1.0 - F.cosine_similarity(current_w.unsqueeze(0), source_w.unsqueeze(0)).item()
                
                if cos_dist < 0.05:
                    weights = sub_sims / sub_sims.sum()
                else:
                    class_densities = active_densities[class_indices][valid_mask]
                    combined_weight = sub_sims / (class_densities + 1e-6)
                    weights = combined_weight / combined_weight.sum()

                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_hpef(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Hypervector Prototype EMA Forking"""
        if getattr(self, 'hpef_slow_prototypes', None) is None:
            self.hpef_slow_prototypes = self.classify.weight.clone().detach()

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

            logits_fast = self.classify(enc_norm)
            logits_slow = F.linear(enc_norm, self.hpef_slow_prototypes)
            
            max_fast, preds_fast = logits_fast.max(dim=1)
            max_slow, preds_slow = logits_slow.max(dim=1)
            predictions = torch.where(max_fast > max_slow, preds_fast, preds_slow)

            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)
                
                slow_w = self.hpef_slow_prototypes[c_id].float()
                updated_slow = (1.0 - effective_lr * 0.1) * slow_w + (effective_lr * 0.1) * self.proto_momentum[c_id].float()
                self.hpef_slow_prototypes[c_id] = F.normalize(updated_slow.unsqueeze(0), dim=1).squeeze(0).to(self.hpef_slow_prototypes.dtype)

            return full_predictions

    def inference_update_csbc(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Confidence-Stratified Batch Correction"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                median_sim = sub_sims.median()
                stratified_weights = torch.where(sub_sims > median_sim, 1.5, 0.5).to(sub_sims.device)
                adjusted_sims = sub_sims * stratified_weights

                weights = adjusted_sims / adjusted_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_gprp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Geometry-Preserving Residual Pull"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                gamma = 0.5 # Minimum separation (cosine distance)
                delta = self.proto_momentum[c_id].float() - current_w
                for j in range(self.num_classes):
                    if j == c_id: continue
                    neighbor_w = self.classify.weight[j].float()
                    sim_current = F.cosine_similarity(current_w.unsqueeze(0), neighbor_w.unsqueeze(0)).item()
                    
                    simulated_update = current_w + effective_lr * delta
                    sim_simulated = F.cosine_similarity(simulated_update.unsqueeze(0), neighbor_w.unsqueeze(0)).item()
                    
                    if (1.0 - sim_simulated) < gamma and sim_simulated > sim_current:
                        proj_scalar = torch.dot(delta, neighbor_w) / (torch.dot(neighbor_w, neighbor_w) + 1e-8)
                        delta = delta - proj_scalar * neighbor_w
                
                updated_weight = current_w + effective_lr * delta
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_modc(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Minority-Only Density Calibration"""
        if getattr(self, 'modc_counts', None) is None:
            self.modc_counts = torch.zeros(self.num_classes, device=self.device)

        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            active_ranges = x_flat[valid_enc_mask, 0].abs()
            active_densities = 1.0 / (active_ranges + 1e-4)

            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            chunk_counts = torch.zeros(self.num_classes, device=self.device)
            unique_classes = torch.unique(predictions)
            
            class_data = []
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue
                    
                sample_encs = enc_norm[class_indices]
                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue
                    
                chunk_counts[c_id] = valid_mask.sum().float()
                class_data.append((c_id, class_indices, valid_mask, sample_encs, sub_sims))
                
            self.modc_counts = 0.9 * self.modc_counts + 0.1 * chunk_counts
            _, top3_idx = torch.topk(self.modc_counts, 3)
            
            for c_id, class_indices, valid_mask, sample_encs, sub_sims in class_data:
                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                
                if c_id in top3_idx:
                    # Density calibration for majority
                    class_densities = active_densities[class_indices][valid_mask]
                    combined_weight = sub_sims / (class_densities + 1e-6)
                    weights = combined_weight / combined_weight.sum()
                else:
                    # Pure Standard Pull for minority
                    weights = sub_sims / sub_sims.sum()

                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_acm(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Asymmetric Confidence Momentum"""
        if getattr(self, 'acm_ema_conf', None) is None:
            self.acm_ema_conf = torch.ones(self.num_classes, device=self.device) * 0.70

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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                mean_sim = sub_sims.mean().item()
                if mean_sim > self.acm_ema_conf[c_id].item():
                    momentum = 0.8
                else:
                    momentum = 0.99

                self.acm_ema_conf[c_id] = 0.9 * self.acm_ema_conf[c_id] + 0.1 * mean_sim

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * mean_sim

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (momentum * self.proto_momentum[c_id] + (1.0 - momentum) * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_pvd(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Prototype Velocity Damping"""
        if getattr(self, 'pvd_velocity', None) is None:
            self.pvd_velocity = torch.zeros_like(self.classify.weight)

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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                
                current_w = self.classify.weight[c_id].float()
                delta = weighted_pull_vector - current_w
                
                velocity = self.pvd_velocity[c_id].float()
                if velocity.norm() > 1e-6:
                    sim = F.cosine_similarity(delta.unsqueeze(0), velocity.unsqueeze(0)).item()
                    if sim < 0:
                        delta = delta * (1.0 + sim)
                
                self.pvd_velocity[c_id] = (0.9 * velocity + 0.1 * delta).to(self.pvd_velocity.dtype)
                
                effective_lr = learning_rate * sub_sims.mean().item()
                
                damped_pull = current_w + delta
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * damped_pull).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_sccp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Sparse Confident Core Pull"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                k = min(10, sample_encs.shape[0])
                top_sims, top_idx = torch.topk(sub_sims, k)
                sample_encs = sample_encs[top_idx]
                sub_sims = top_sims

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_cmop(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Confident Minority Oracle Pull"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            active_labels = None
            if oracle_labels is not None:
                active_labels = oracle_labels.view(-1)[valid_enc_mask]

            unique_classes = torch.unique(predictions)
            
            class_data = []
            class_counts = []
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue
                    
                sample_encs = enc_norm[class_indices]
                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue
                
                count = valid_mask.sum().item()
                class_counts.append((count, c_id))
                class_data.append((c_id, class_indices, valid_mask, sample_encs, sub_sims))
                
            class_counts.sort() # Sort by count
            minority_classes = [c_id for count, c_id in class_counts[:3]]
            
            for c_id, class_indices, valid_mask, sample_encs, sub_sims in class_data:
                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                class_indices = class_indices[valid_mask]
                
                if c_id in minority_classes and active_labels is not None:
                    max_idx = torch.argmax(sub_sims).item()
                    global_idx = class_indices[max_idx].item()
                    gt_label = active_labels[global_idx].item()
                    
                    if gt_label == c_id:
                        pull_vector = sample_encs[max_idx]
                        effective_lr = learning_rate * 3.0 * sub_sims[max_idx].item()
                    else:
                        continue # Oracle rejects, discard update for this class entirely for this chunk? 
                        # Actually let's just not update the minority class if the best is wrong.
                else:
                    weights = sub_sims / sub_sims.sum()
                    pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                    effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_pdi(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Prototype Drift Intervention"""
        if getattr(self, 'pdi_alpha', None) is None:
            self.pdi_alpha = torch.ones(self.num_classes, device=self.device) * 0.9
            
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            active_labels = None
            if oracle_labels is not None:
                active_labels = oracle_labels.view(-1)[valid_enc_mask]

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                valid_class_indices = class_indices[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                
                self.pdi_alpha[c_id] = max(0.9, self.pdi_alpha[c_id].item() - 0.01)
                alpha = self.pdi_alpha[c_id].item()
                
                simulated_momentum = (alpha * self.proto_momentum[c_id] + (1.0 - alpha) * weighted_pull_vector).to(self.proto_momentum.dtype)
                simulated_updated_weight = (1.0 - effective_lr) * current_w + effective_lr * simulated_momentum.float()
                simulated_w_norm = F.normalize(simulated_updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
                drift = 1.0 - F.cosine_similarity(current_w.unsqueeze(0), simulated_w_norm.unsqueeze(0)).item()
                
                if drift > 0.005 and active_labels is not None: 
                    k = min(5, sample_encs.shape[0])
                    _, top_idx = torch.topk(sub_sims, k)
                    global_indices = valid_class_indices[top_idx]
                    
                    gt_labels = active_labels[global_indices]
                    correct_count = (gt_labels == c_id).sum().item()
                    
                    if correct_count < k / 2.0:
                        self.pdi_alpha[c_id] = 0.99
                        continue 
                        
                self.proto_momentum[c_id] = simulated_momentum
                self.classify.weight[c_id] = simulated_w_norm.to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_cbot(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Class Boundary Oracle Triangulation"""
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            active_labels = None
            if oracle_labels is not None:
                active_labels = oracle_labels.view(-1)[valid_enc_mask]

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            if active_labels is not None:
                pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (11, 12)] 
                for p1, p2 in pairs:
                    if p1 >= self.num_classes or p2 >= self.num_classes: continue
                    
                    sim1 = chunk_logits[:, p1]
                    sim2 = chunk_logits[:, p2]
                    
                    score = torch.abs(sim1 - sim2) - 0.5 * (sim1 + sim2)
                    idx = torch.argmin(score).item()
                    
                    gt_label = active_labels[idx].item()
                    if gt_label == p1 or gt_label == p2:
                        correct_p = gt_label
                        wrong_p = p2 if gt_label == p1 else p1
                        
                        sample_enc = enc_norm[idx].float()
                        wrong_w = self.classify.weight[wrong_p].float()
                        
                        proj = sample_enc - torch.dot(sample_enc, wrong_w) * wrong_w
                        proj = F.normalize(proj.unsqueeze(0), dim=1).squeeze(0)
                        
                        current_w = self.classify.weight[correct_p].float()
                        effective_lr = learning_rate * 3.0
                        updated_weight = (1.0 - effective_lr) * current_w + effective_lr * proj
                        self.classify.weight[correct_p] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)
                        
            return full_predictions

    def inference_update_taoc(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None):
        """Temporal Anchor Oracle Correction"""
        if getattr(self, 'taoc_buffer', None) is None:
            self.taoc_buffer = {c: [] for c in range(self.num_classes)}
            self.taoc_chunk_counter = 0
            
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

            chunk_logits = self.classify(enc_norm)
            predictions = chunk_logits.argmax(dim=1)
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=predictions.dtype)
            full_predictions[valid_enc_mask] = predictions

            active_labels = None
            if oracle_labels is not None:
                active_labels = oracle_labels.view(-1)[valid_enc_mask]

            unique_classes = torch.unique(predictions)
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id)
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if len(class_indices) == 0:
                    continue

                sample_encs = enc_norm[class_indices]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                valid_class_indices = class_indices[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_w = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                
                updated_weight = (1.0 - effective_lr) * current_w + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

                if active_labels is not None:
                    cur_gt_cpu = active_labels[valid_class_indices].cpu().numpy()
                    cur_sims_cpu = sub_sims.cpu().numpy()
                    cur_encs = sample_encs.detach()
                    
                    new_items = [{'enc': cur_encs[i], 'sim': cur_sims_cpu[i], 'gt': cur_gt_cpu[i]} for i in range(len(cur_gt_cpu))]
                    self.taoc_buffer[c_id].extend(new_items)
                    
                    if len(self.taoc_buffer[c_id]) > 50:
                        self.taoc_buffer[c_id] = self.taoc_buffer[c_id][-50:]
            
            self.taoc_chunk_counter += 1
            if self.taoc_chunk_counter >= 10 and active_labels is not None:
                self.taoc_chunk_counter = 0
                for c_id in range(self.num_classes):
                    if len(self.taoc_buffer[c_id]) == 0:
                        continue
                    
                    self.taoc_buffer[c_id].sort(key=lambda x: x['sim'])
                    
                    to_remove = []
                    for i in range(min(3, len(self.taoc_buffer[c_id]))):
                        item = self.taoc_buffer[c_id][i]
                        if item['gt'] != c_id:
                            bad_h = item['enc'].float()
                            current_w = self.classify.weight[c_id].float()
                            
                            unpull = current_w - 0.1 * learning_rate * bad_h
                            self.classify.weight[c_id] = F.normalize(unpull.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)
                            to_remove.append(i)
                            
                    for i in sorted(to_remove, reverse=True):
                        self.taoc_buffer[c_id].pop(i)

            return full_predictions

    def inference_update_ttaug(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Hypervector Bundling via Test-Time Augmentation (TTAug)"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            
            # Generate 3 augmented views for Multi-Scale Constructive Interference
            x_aug_rot = torch.roll(x, shifts=14, dims=3) # Spatially rotated (+5 degrees yaw, ~14 pixels on 1024 width)
            x_aug_scale = x * 0.95                       # Scaled slightly (95% size)
            
            enc_rot, _, _ = self.encode(x_aug_rot)
            enc_scale, _, _ = self.encode(x_aug_scale)
            
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            active_enc_rot = enc_rot[valid_enc_mask]
            active_enc_scale = enc_scale[valid_enc_mask]
            
            # Bundle hypervectors (H_bundled = H_A + H_B + H_C)
            bundled_enc = active_enc + active_enc_rot + active_enc_scale
            enc_norm = F.normalize(bundled_enc)
            
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            num_active = enc_norm.shape[0]
            curr_chunk_size = num_active if chunk_size == -1 else chunk_size

            all_predictions = []
            all_update_masks = []

            for i in range(0, num_active, curr_chunk_size):
                chunk_enc_norm = enc_norm[i : i + curr_chunk_size]
                chunk_logits = self.classify(chunk_enc_norm)
                chunk_preds = torch.argmax(chunk_logits, dim=1)
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

    def inference_update_gplp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Graph-Laplacian Label Propagation"""
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
                
            # Raw cosine similarity against HDC prototypes
            prototypes = F.normalize(self.classify.weight)
            S = enc_norm @ prototypes.T  # Shape: (num_active, num_classes)
            
            if proj_xyz is not None:
                # Extract active xyz points
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                # KNN graph in physical space
                # To prevent OOM, we can do cdist in chunks if needed, but active points are usually <30k
                # A 30k x 30k float32 matrix is ~3.6GB. Let's do it safely.
                num_active = active_xyz.shape[0]
                K = 5
                iterations = 3
                
                # Construct graph step-by-step to save memory
                # We will just compute top-K for each row
                topk_indices = []
                chunk_s = 5000
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    _, knn_idx = dist.topk(K + 1, largest=False)
                    topk_indices.append(knn_idx[:, 1:]) # Exclude self
                knn_idx = torch.cat(topk_indices, dim=0) # (num_active, K)
                
                # Propagate scores (Graph Laplacian smoothing)
                for _ in range(iterations):
                    # For each node, new score is average of its own and its neighbors
                    neighbor_scores = S[knn_idx] # (num_active, K, num_classes)
                    S = 0.5 * S + 0.5 * neighbor_scores.mean(dim=1)
            
            predictions = S.argmax(dim=1)
            
            # Now we use the smoothed similarity (or recompute distance to pulled prototypes)
            # Standard standard pull logic follows:
            all_update_masks = []
            selected_proto = prototypes[predictions]
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            # But wait, we should use the smoothed S for thresholding? 
            # The prompt says: "If a pedestrian's torso is highly confident (0.85) but legs are corrupted (0.40), the graph propagation will allow the torso's confidence to 'bleed' down... raising the legs to 0.75. Now, the entire pedestrian crosses the threshold, pulling the prototype cohesively"
            # So the pull criteria is based on the smoothed similarity S!
            smoothed_sims = S.gather(1, predictions.unsqueeze(1)).squeeze(1)
            distances = (1.0 - smoothed_sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = predictions

            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(predictions[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id) & update_mask
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

    def inference_update_with_subcluster_pull(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, subcluster_lr=0.005, thresholds=(0.45, 0.80)):
        """Temp Function for testing. Like inference_update, but also pulls the nearest subcluster toward high-confidence samples."""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            if not torch.any(valid_enc_mask):
                return torch.zeros(enc.shape[0], device=self.device, dtype=torch.long)

            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            chunk_logits = self.classify(enc_norm)
            predictions = torch.argmax(chunk_logits, dim=1)

            if self.subcluster_type == 'bipolar':
                proto_binary = torch.sign(self.classify.weight)
                enc_binary = torch.sign(active_enc)
                selected_proto = proto_binary[predictions]
                sims = torch.sum(enc_binary * selected_proto, dim=1) / self.hd_dim
            else:
                selected_proto = F.normalize(self.classify.weight[predictions])
                sims = torch.sum(enc_norm * selected_proto, dim=1)

            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta

            unique_classes = torch.unique(predictions[update_mask])
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id) & update_mask
                sample_encs = enc_norm[class_mask]

                if self.subcluster_type == 'bipolar':
                    target_encs = torch.sign(active_enc[class_mask])
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(target_encs, c_id, distance_sensitivity)
                else:
                    sub_sims, sub_indices = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                sub_indices = sub_indices[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                eff_lr = learning_rate * sub_sims.mean().item()
                current_w = self.classify.weight[c_id]
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull)
                updated_w = (1.0 - eff_lr) * current_w + eff_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_w.unsqueeze(0), dim=1).squeeze(0)

                unique_subs, inv_idx = torch.unique(sub_indices, return_inverse=True)
                for i, abs_idx in enumerate(unique_subs.tolist()):
                    member_mask = inv_idx == i
                    member_encs = sample_encs[member_mask]
                    member_sims = sub_sims[member_mask]
                    w = member_sims / member_sims.sum()
                    pull_vec = (member_encs * w.unsqueeze(1)).sum(dim=0)
                    current_sub = self.subclusters.data[abs_idx].float()
                    updated_sub = (1.0 - subcluster_lr) * current_sub + subcluster_lr * pull_vec
                    if self.subcluster_type == 'bipolar':
                        updated_sub = torch.sign(updated_sub)
                        updated_sub[updated_sub == 0] = -1.0
                    self.subclusters.data[abs_idx] = F.normalize(
                        updated_sub.unsqueeze(0), dim=1).squeeze(0)

            full_predictions = torch.zeros(enc.shape[0], device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = predictions
            return full_predictions
        
    def entropy_minimization_step(self, x, optimizer, temperature=1.0):
        """
        One gradient step of entropy minimization on unlabelled target batch x for self.net

        temperature: float  softens logits before entropy (>1 = softer)
        """
        self.net.train()
        self.classify.eval()

        with torch.amp.autocast('cuda', enabled=True):
            feat = self.net(x, only_feat=True)

        feat = feat.permute(0, 2, 3, 1).reshape(-1, 128).float()
        sample_hv = torch.zeros(feat.shape[0], self.hd_dim, device=self.device, dtype=feat.dtype)

        if self.hd_encoder == 'rp':
            if feat.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(feat.dtype).to(self.device)
            sample_hv = self.projection(feat)
        elif self.hd_encoder == 'nonlinear':
            sample_hv = self.nonlinear_projection(feat)
        else:
            sample_hv = feat

        if self.classify.weight.dtype != sample_hv.dtype:
            self.classify = self.classify.to(sample_hv.dtype)

        logits = F.linear(F.normalize(sample_hv), self.classify.weight.detach()) / temperature

        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()

        return entropy.item()

    def get_accuracy(self, x, labels):
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)

            logits = self.get_predictions(enc)
            predictions = torch.argmax(logits, dim=1)
        
            confidences = torch.softmax(logits, dim=1)
            max_confidences, _ = torch.max(confidences, dim=1)

            batch_size = x.shape[0]
            h, w = x.shape[-2:]
            predictions_2d = predictions.reshape(batch_size, h, w)
            confidence_map = max_confidences.reshape(batch_size, h, w)

            if labels.dim() == 4:
                labels = labels.squeeze(1)

            pred_flat = predictions_2d.flatten()
            label_flat = labels.flatten()

            correct = (pred_flat == label_flat).sum().item()
            total = len(pred_flat)
            accuracy = correct / total
            unique_classes = torch.unique(label_flat)
            class_accuracies = {}

            for cls in unique_classes:
                if cls.item() == 255:
                    continue
                cls_mask = label_flat == cls
                if torch.any(cls_mask):
                    cls_correct = (pred_flat[cls_mask] == cls).sum().item()
                    cls_total = cls_mask.sum().item()
                    class_accuracies[cls.item()] = cls_correct / cls_total if cls_total > 0 else 0.0

        return accuracy, confidence_map, class_accuracies

    def inference_update_gvgb(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Geometric Variance-Gated Bundling (GVGB)"""
        self.eval()
        with torch.no_grad():
            x_aug_rot = torch.roll(x, shifts=14, dims=3)
            x_aug_scale = x * 0.95
            
            enc_A, _, _ = self.encode(x)
            enc_B, _, _ = self.encode(x_aug_rot)
            enc_C, _, _ = self.encode(x_aug_scale)
            
            num_total_samples = enc_A.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc_A = enc_A[valid_enc_mask]
            active_enc_B = enc_B[valid_enc_mask]
            active_enc_C = enc_C[valid_enc_mask]
            
            norm_A = F.normalize(active_enc_A)
            norm_B = F.normalize(active_enc_B)
            norm_C = F.normalize(active_enc_C)
            
            sim_AB = torch.sum(norm_A * norm_B, dim=1)
            sim_AC = torch.sum(norm_A * norm_C, dim=1)
            
            variance_mask = (sim_AB >= 0.40) & (sim_AC >= 0.40)
            
            # Use only valid points for bundling
            valid_enc_A = active_enc_A[variance_mask]
            valid_enc_B = active_enc_B[variance_mask]
            valid_enc_C = active_enc_C[variance_mask]
            
            bundled_enc = valid_enc_A + valid_enc_B + valid_enc_C
            enc_norm = F.normalize(bundled_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                
            chunk_logits = self.classify(enc_norm)
            preds = torch.argmax(chunk_logits, dim=1)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_predictions = torch.zeros(active_enc_A.shape[0], device=self.device, dtype=torch.long)
            active_predictions[~variance_mask] = torch.argmax(self.classify(norm_A[~variance_mask].to(self.classify.weight.dtype)), dim=1)
            active_predictions[variance_mask] = preds
            
            full_predictions[valid_enc_mask] = active_predictions

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

    def inference_update_dabp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Density-Aware Bundled Pull (DABP)"""
        self.eval()
        with torch.no_grad():
            x_aug_rot = torch.roll(x, shifts=14, dims=3)
            x_aug_scale = x * 0.95
            
            enc_A, _, _ = self.encode(x)
            enc_B, _, _ = self.encode(x_aug_rot)
            enc_C, _, _ = self.encode(x_aug_scale)
            
            num_total_samples = enc_A.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            active_ranges = x_flat[valid_enc_mask, 0] 
            
            active_enc_A = enc_A[valid_enc_mask]
            active_enc_B = enc_B[valid_enc_mask]
            active_enc_C = enc_C[valid_enc_mask]
            
            bundled_enc = active_enc_A + active_enc_B + active_enc_C
            enc_norm = F.normalize(bundled_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                
            chunk_logits = self.classify(enc_norm)
            preds = torch.argmax(chunk_logits, dim=1)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

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
                
                # Density-Calibrated Pull Logic (from DCSP)
                sample_ranges = active_ranges[class_indices][valid_mask].abs()
                range_scale = sample_ranges / (sample_ranges.max() + 1e-4)
                
                combined_weight = sub_sims * range_scale
                weights = combined_weight / combined_weight.sum()

                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

def set_dense_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='bipolar'):
    return DensityModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type=subcluster_type)
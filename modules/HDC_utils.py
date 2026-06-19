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
            # # Pick by the wrong and keep the PERCENTAGE
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select
            # selected_indices = torch.randperm(x.shape[0], device=x.device)[:num_samples]
            # print("selected_indices", selected_indices.shape)  # e.g., torch.Size([1638])
            # print("x", x.shape)  # e.g., torch.Size([1638])
            # print("x[selected_indices]", x[selected_indices[0]])  # e.g., torch.Size([1638, 128])

            # # print("num_samples", num_samples)  # e.g., 32768 * 0.05 = 1638
            # # print("wrong_indices", wrong_indices.shape)
            # # print("is_wrong", is_wrong.shape)  # e.g., torch.Size([32768])

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
        """Process a single class to generate its subclusters."""
        if len(class_emb_np) == 0:
            return []
        
        print(f"  Running mean shift on {len(class_emb_np)} samples...")
        cluster_centers = mean_shift_binary(
            X=class_emb_np,
            bandwidth=bandwidth,
            quantile=self.quantile,
            bandwidth_multiplier=self.mult,
            dedup_scale=self.dedup
        )
        if self.subcluster_type == "bipolar":
            cluster_centers = np.sign(cluster_centers)
        
        num_clusters_found = len(cluster_centers)
        print(f"  Found {num_clusters_found} clusters")

        subclusters = []
        if num_clusters_found <= num_sub_per_cluster:
            for center in cluster_centers:
                center_tensor = torch.tensor(center, device='cpu', dtype=torch.float32)
                subclusters.append(center_tensor)
        else:
            center_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
            fps_indices = self._farthest_point_sample(center_tensor, num_sub_per_cluster)
            for idx in fps_indices.tolist():
                center = torch.tensor(cluster_centers[idx], device='cpu', dtype=torch.float32)
                subclusters.append(center)

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
        self.train()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]

            valid_enc_mask = torch.any(enc != 0, dim=1) # ignore background from updates
            
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
        self.train()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = torch.any(enc != 0, dim=1)
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

    def inference_update_cnp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], push_ratio=0.5):
        """Contrastive Negative Push Update"""
        self.train()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = torch.any(enc != 0, dim=1)
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                
            chunk_logits = self.classify(enc_norm)
            
            top2_logits, top2_indices = torch.topk(chunk_logits, 2, dim=1)
            predictions = top2_indices[:, 0]
            runner_ups = top2_indices[:, 1]
            
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

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]
                batch_runner_ups = runner_ups[class_indices][valid_mask]

                weights_sample = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights_sample.unsqueeze(1)).sum(dim=0).float()
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)
                
                unique_runners = torch.unique(batch_runner_ups)
                for r_id in unique_runners:
                    r_id = r_id.item()
                    r_mask = batch_runner_ups == r_id
                    r_samples = sample_encs[r_mask]
                    r_weights = sub_sims[r_mask]
                    r_weights = r_weights / (r_weights.sum() + 1e-8)
                    weighted_push_vector = (r_samples * r_weights.unsqueeze(1)).sum(dim=0).float()
                    
                    r_effective_lr = learning_rate * push_ratio * sub_sims[r_mask].mean().item()
                    r_current_weight = self.classify.weight[r_id].float()
                    r_updated_weight = r_current_weight - r_effective_lr * weighted_push_vector
                    self.classify.weight[r_id] = F.normalize(r_updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_dat(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80]):
        """Dynamic Adaptive Thresholding Update"""
        self.train()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = torch.any(enc != 0, dim=1)
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
            
            batch_mu = max_sims.mean()
            batch_std = max_sims.std()
            dynamic_threshold = torch.clamp(batch_mu + 1.0 * batch_std, min=thresholds[0], max=thresholds[1]).item()
            
            update_mask = max_sims > dynamic_threshold
            
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

                valid_mask = sub_sims > dynamic_threshold
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights_sample = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights_sample.unsqueeze(1)).sum(dim=0).float()
                
                effective_lr = learning_rate * 2.0 * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id].float()
                self.proto_momentum[c_id] = (0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector).to(self.proto_momentum.dtype)
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id].float()
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions

    def inference_update_cdsd(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80]):
        """Confidence-Decayed Subcluster Distillation Update"""
        self.train()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = torch.any(enc != 0, dim=1)
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
                    
                    abs_idx = c_id * self.num_subclusters + sub_idx
                    current_sub = self.subclusters.data[abs_idx].float()
                    
                    subcluster_lr = learning_rate * s_sims.mean().item()
                    updated_sub = (1.0 - subcluster_lr) * current_sub + subcluster_lr * pull_vec
                    
                    if self.subcluster_type == 'continuous':
                        updated_sub = F.normalize(updated_sub.unsqueeze(0), dim=1).squeeze(0)
                    else:
                        updated_sub = torch.sign(updated_sub)
                        
                    self.subclusters.data[abs_idx] = updated_sub.to(self.subclusters.dtype)

                start_idx = c_id * self.num_subclusters
                end_idx = start_idx + self.num_subclusters
                class_subclusters = self.subclusters.data[start_idx:end_idx].float()
                
                distilled_prototype = class_subclusters.mean(dim=0)
                self.classify.weight[c_id] = F.normalize(distilled_prototype.unsqueeze(0), dim=1).squeeze(0).to(self.classify.weight.dtype)

            return full_predictions
        
    def inference_update_with_subcluster_pull(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, subcluster_lr=0.005, thresholds=(0.45, 0.80)):
        """Temp Function for testing. Like inference_update, but also pulls the nearest subcluster toward high-confidence samples."""
        self.train()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            valid_enc_mask = torch.any(enc != 0, dim=1)
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

def set_dense_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='bipolar'):
    return DensityModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type=subcluster_type)
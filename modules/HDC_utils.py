from torchhd import functional
from torchhd import embeddings

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, max_subclusters = 5, subcluster_type="continuous"):
        super(DensityModel, self).__init__()

        self.device = device

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
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            torch.manual_seed(42) # setting fixed seed for projection initialization (removes saved model randomness)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

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

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.amp.autocast('cuda', enabled=True):
            x = self.net(x, True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

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
    
    def init_subclusters(self, dataloader, bandwidth=None, max_samples_per_class=5000, sampling_strategy='diverse'):
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
                    proj_in = proj_in.to(self.device)
                    proj_labels = proj_labels.to(self.device).flatten()
                    enc, _, _ = self.encode(proj_in)
                    class_mask = proj_labels == class_id
                    
                    if torch.any(class_mask):
                        class_enc = enc[class_mask].cpu().half()
                        class_embeddings.append(class_enc)
                        batch_indices.extend([batch_idx] * class_enc.shape[0])
                        total_samples += class_enc.shape[0]
                    
                    del proj_in, proj_labels
                    self._clear_memory()
                    
                    print(f"  Batch {batch_idx}: collected {total_samples} samples so far")

                    if total_samples >= MAX_SAMPLES: # collect extra for better sampling
                        break
            
            if not class_embeddings:
                print(f"  No data for class {class_id}, skipping")
                continue
            
            class_emb_cpu = torch.cat(class_embeddings, dim=0)

            if len(class_emb_cpu) > MAX_SAMPLES:
                indices = torch.randperm(len(class_emb_cpu))[:MAX_SAMPLES]
                class_emb_cpu = class_emb_cpu[indices]
            
            batch_indices = torch.tensor(batch_indices[:len(class_emb_cpu)])
            
            class_emb_cpu = torch.cat(class_embeddings, dim=0)
            batch_indices = torch.tensor(batch_indices)

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
        cluster_centers = np.sign(cluster_centers)
        
        num_clusters_found = len(cluster_centers)
        print(f"  Found {num_clusters_found} clusters")

        subclusters = []
        if num_clusters_found <= num_sub_per_cluster:
            for center in cluster_centers:
                center_tensor = torch.tensor(center, device='cpu', dtype=torch.float16)
                subclusters.append(center_tensor)
        else:
            indices = np.random.choice(num_clusters_found, num_sub_per_cluster, replace=False)
            for idx in indices:
                center = torch.tensor(cluster_centers[idx], device='cpu', dtype=torch.float16)
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
                    batch = torch.stack([
                        self._make_bipolar(c.to(self.device)) if c.device.type == 'cpu' 
                        else self._make_bipolar(c) 
                        for c in centers_list[i:end_idx]
                    ])
                    assert torch.all(torch.abs(batch) == 1), f"Subclusters must be bipolar! Got values: {torch.unique(batch)}"
                elif self.subcluster_type == 'continuous':
                    batch = torch.stack([
                        c.to(self.device) if c.device.type == 'cpu' else c 
                        for c in centers_list[i:end_idx]
                    ])
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
        Get maximum similarity [0,1] to subclusters using Hamming distance.
        distance_sensitivity introduces a power law which scales with similarity = base_similarity ^ (1 / distance_sensitivity)
        """
        mask = self.subcluster_to_class == class_id
        relevant_subclusters = self.subclusters[mask]
        
        if len(relevant_subclusters) == 0:
            return torch.zeros(enc.shape[0], device=enc.device), None

        if self.subcluster_type == 'bipolar':
            enc_binary = torch.sign(enc).to(dtype=self.subclusters.dtype)
            hd_dim = enc_binary.shape[1]
            
            dot_products = torch.matmul(enc_binary, relevant_subclusters.T)
            base_similarity = (dot_products + hd_dim) / (2 * hd_dim)
        elif self.subcluster_type == 'continuous':
            enc_norm = F.normalize(enc)
            base_similarity = torch.matmul(enc_norm, relevant_subclusters.T)
            base_similarity = (base_similarity + 1) / 2        
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

    def inference_update(self, x, beta=0.0, distance_sensitivity=1.0, learning_rate=1.0):
        """
        Inference-only update with distance-aware gating.
        Works with both bipolar and continuous subclusters.
        """
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            enc_norm = F.normalize(enc)

            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            logits = self.classify(enc_norm)
            predictions = torch.argmax(logits, dim=1)

            if self.subcluster_type == 'bipolar':
                enc_binary = torch.sign(enc)
                proto_binary = torch.sign(self.classify.weight)
                selected_proto = proto_binary[predictions]
                hd_dim = enc_binary.shape[1]
                similarities = torch.sum(enc_binary * selected_proto, dim=1) / hd_dim
            elif self.subcluster_type == 'continuous':
                selected_proto = F.normalize(self.classify.weight[predictions])
                similarities = torch.sum(enc_norm * selected_proto, dim=1)
            else:
                raise ValueError(f"Unknown subcluster_type: {self.subcluster_type}")

            distances = (1.0 - similarities) / 2.0
            update_mask = distances > beta

            if not torch.any(update_mask):
                return predictions
            enc_norm = enc_norm[update_mask].float()
            predictions = predictions[update_mask]

            if self.subcluster_type == 'bipolar':
                enc_binary = torch.sign(enc)[update_mask].float()

            unique_classes = torch.unique(predictions)

            for class_id in unique_classes:
                class_mask = predictions == class_id
                class_enc_norm = enc_norm[class_mask]

                if self.subcluster_type == 'bipolar':
                    class_enc_binary = enc_binary[class_mask]
                    subcluster_sims, _ = self.get_max_subcluster_similarity(class_enc_binary, class_id, distance_sensitivity=distance_sensitivity)
                elif self.subcluster_type == 'continuous':
                    subcluster_sims, _ = self.get_max_subcluster_similarity(class_enc_norm, class_id, distance_sensitivity=distance_sensitivity)

                scaled = class_enc_norm * subcluster_sims.unsqueeze(1) * learning_rate

                update = scaled.sum(dim=0)

                self.classify.weight[class_id] += update
                self.classify.weight[class_id] = F.normalize(self.classify.weight[class_id:class_id+1], dim=1)[0]

            return predictions

    def chunked_inference_update(self, x, beta=0, distance_sensitivity=5.0, learning_rate=1.0, chunk_size=1000, verbose=False):
        """
        Inference with updates scaled by subcluster distance.
        Not updated as frequently as inference_update, may be wrong.
        """
        with torch.no_grad():
            self.classify_weights.data.copy_(self.classify.weight.data)
            
            if verbose:
                print(f"\n[DIAGNOSTIC] Before update:")
                print(f"  classify.weight norms: min={torch.norm(self.classify.weight, dim=1).min():.6f}, max={torch.norm(self.classify.weight, dim=1).max():.6f}")
                print(f"  classify_weights norms: min={torch.norm(self.classify_weights, dim=1).min():.6f}, max={torch.norm(self.classify_weights, dim=1).max():.6f}")
            
            enc, _, _ = self.encode(x)

            enc_binary = torch.sign(enc)
            zero_mask = enc_binary == 0
            if torch.any(zero_mask):
                enc_binary[zero_mask] = -1.0

            enc_normalized = F.normalize(enc)

            if enc_normalized.dtype != self.classify.weight.dtype:
                enc_normalized = enc_normalized.to(self.classify.weight.dtype)

            logits = self.classify(enc_normalized)
            predictions = torch.argmax(logits, dim=1)

            prototypes_binary = torch.sign(self.classify.weight)
            selected_prototypes = prototypes_binary[predictions]
            hd_dim = enc_binary.shape[1]
            similarities = torch.sum(enc_binary * selected_prototypes, dim=1) / hd_dim
            distances = (1 - similarities) / 2
            mask = distances > beta

            if verbose:
                print(f"  Samples needing update: {mask.sum().item()}/{len(mask)} ({100*mask.float().mean():.1f}%)")
                print(f"  Distance stats: min={distances.min():.4f}, max={distances.max():.4f}, mean={distances.mean():.4f}")

            if not torch.any(mask):
                return predictions

            distant_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            num_distant = len(distant_indices)
            
            update_magnitudes = []
            subcluster_sim_stats = []

            for chunk_start in range(0, num_distant, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_distant)
                chunk_indices = distant_indices[chunk_start:chunk_end]

                chunk_enc_norm = enc_normalized[chunk_indices]
                chunk_enc_binary = enc_binary[chunk_indices]
                chunk_predictions = predictions[chunk_indices]
                
                unique_classes = torch.unique(chunk_predictions)
                
                for class_id in unique_classes:
                    class_mask = chunk_predictions == class_id
                    class_enc_norm = chunk_enc_norm[class_mask]
                    class_enc_binary = chunk_enc_binary[class_mask]

                    # Get subcluster similarity weights using bipolar vectors
                    subcluster_sims, _ = self.get_max_subcluster_similarity(
                        class_enc_binary, class_id, distance_sensitivity=distance_sensitivity
                    )
                    
                    if verbose:
                        subcluster_sim_stats.append({
                            'class': class_id.item(),
                            'min': subcluster_sims.min().item(),
                            'max': subcluster_sims.max().item(),
                            'mean': subcluster_sims.mean().item()
                        })

                    # Scale the NORMALIZED updates by subcluster similarity
                    # This matches retrain() where we add normalized hypervectors
                    scaled_samples = class_enc_norm * subcluster_sims.unsqueeze(1) * learning_rate
                    
                    total_update = scaled_samples.sum(dim=0)
                    
                    if verbose:
                        update_magnitudes.append({
                            'class': class_id.item(),
                            'magnitude': torch.norm(total_update).item(),
                            'num_samples': len(class_enc_norm)
                        })

                    self.classify_weights[class_id] += total_update

                    del class_enc_norm, class_enc_binary, subcluster_sims, scaled_samples, total_update

                del chunk_enc_norm, chunk_enc_binary, chunk_predictions
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if verbose:
                print(f"\n  Subcluster similarity stats:")
                for stat in subcluster_sim_stats[:5]:
                    print(f"    Class {stat['class']}: min={stat['min']:.4f}, max={stat['max']:.4f}, mean={stat['mean']:.4f}")
                
                print(f"\n  Update magnitudes:")
                for stat in update_magnitudes[:5]:
                    print(f"    Class {stat['class']}: magnitude={stat['magnitude']:.4f}, samples={stat['num_samples']}")

            unique_updated_classes = torch.unique(predictions[distant_indices])

            for class_id in unique_updated_classes:
                if verbose and class_id == unique_updated_classes[0]:
                    old_weight = self.classify.weight[class_id].clone()
                
                self.classify.weight[class_id] = F.normalize(
                    self.classify_weights[class_id:class_id+1], dim=1
                )[0]
                
                if verbose and class_id == unique_updated_classes[0]:
                    change = torch.norm(self.classify.weight[class_id] - old_weight).item()
                    print(f"\n  Example class {class_id.item()} weight change: {change:.6f}")

            if verbose:
                print(f"\n[DIAGNOSTIC] After update:")
                print(f"  classify.weight norms: min={torch.norm(self.classify.weight, dim=1).min():.6f}, max={torch.norm(self.classify.weight, dim=1).max():.6f}")
                final_norms = torch.norm(self.classify.weight, dim=1)
                max_dev = torch.abs(final_norms - 1.0).max().item()
                print(f"  Max deviation from unit norm: {max_dev:.6f}")

            del enc_normalized, enc_binary, prototypes_binary, selected_prototypes
            del similarities, distances, mask, distant_indices
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return predictions
            
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

def set_dense_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='continuous'):
    return DensityModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type=subcluster_type)
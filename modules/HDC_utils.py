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

def set_dense_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, hd_dim=10000):
    return DensityModel(ARCH, modeldir, num_classes, device, hd_dim=hd_dim)

class DensityModel(nn.Module):
    def __init__(self, ARCH, modeldir, num_classes, device, hd_dim=10000, max_subclusters = 5):
        super(DensityModel, self).__init__()

        self.device = device

        self.num_classes = num_classes
        self.hd_dim = hd_dim
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

        self.projection = embeddings.Projection(self.input_dim, self.hd_dim)
        self.projection = self.projection.to(self.device)

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False, device=self.device)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)

        # density subcluster initialization
        self.num_subclusters = max_subclusters
        self.subclusters = nn.Parameter(torch.zeros(self.num_classes * self.num_subclusters, self.hd_dim, device=self.device))
        self.subclusters.data.fill_(0.0)

        self.subcluster_to_class = torch.repeat_interleave(torch.arange(self.num_classes, device=self.device), self.num_subclusters)

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.amp.autocast(self.device.type, enabled=True):
            x = self.net(x, True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)
        if PERCENTAGE is not None:
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)

            if wrong_indices.numel() >= num_samples:
                selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                is_wrong[selected_indices] = False
            else:
                non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                remaining = num_samples - wrong_indices.numel()
                fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]

                selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                is_wrong[selected_indices] = False # Mark the selected indices as used

            selected_indices, _ = selected_indices.sort()
            x = x[selected_indices]
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"
        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if x.dtype != self.projection.weight.dtype:
            self.projection = self.projection.to(x.dtype).to(self.device)
        sample_hv[:, mask] = self.projection(x)[:, mask]

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv, selected_indices, is_wrong
        
    def encode_chunked(self, x, mask=None, PERCENTAGE=None, is_wrong=None, chunk_size=1000, projection_chunk_size=5000):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        self._clear_memory()

        batch_size = x.shape[0]
        all_embeddings = []
        
        for i in range(0, batch_size, chunk_size):
            end_i = min(i + chunk_size, batch_size)
            x_chunk = x[i:end_i]
            
            with torch.amp.autocast(self.device.type, enabled=True):
                emb_chunk = self.net(x_chunk, True)
            
            emb_chunk = emb_chunk.permute(0, 2, 3, 1)
            emb_chunk = emb_chunk.reshape(-1, 128)
            all_embeddings.append(emb_chunk)
            
            del emb_chunk
            self._clear_memory()
        
        x = torch.cat(all_embeddings, dim=0)
        del all_embeddings
        x = x.to(self.device)

        if PERCENTAGE is not None:
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)

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
            selected_indices = torch.arange(x.shape[0], device=x.device)
        
        total_points = x.shape[0]
        sample_hv = torch.zeros((total_points, self.hd_dim), device=self.device, dtype=torch.float16)

        for proj_start in range(0, total_points, projection_chunk_size):
            proj_end = min(proj_start + projection_chunk_size, total_points)
            x_chunk = x[proj_start:proj_end]

            for inner_start in range(0, x_chunk.shape[0], 1000):
                inner_end = min(inner_start + 1000, x_chunk.shape[0])
                x_inner = x_chunk[inner_start:inner_end]
                
                with torch.amp.autocast(self.device.type, enabled=True):
                    projected_inner = self.projection(x_inner)

                projected_inner[:, mask] = self._quantize_tiny_chunks(projected_inner[:, mask])
                sample_hv[proj_start + inner_start:proj_start + inner_end, mask] = projected_inner[:, mask]
                
                del projected_inner, x_inner
                self._clear_memory()
            
            del x_chunk
            self._clear_memory()

        del x
        self._clear_memory()
        
        return sample_hv, selected_indices, is_wrong

    def _quantize_tiny_chunks(self, tensor, max_chunk_elements=100000):
        """Quantize in very small chunks to minimize memory."""
        if tensor.numel() <= max_chunk_elements:
            return torch.where(tensor > 0, 1, -1).to(tensor.dtype)
        
        result = torch.empty_like(tensor)
        rows = tensor.shape[0]
        cols = tensor.shape[1]

        chunk_rows = max(1, max_chunk_elements // cols)
        
        for i in range(0, rows, chunk_rows):
            end_i = min(i + chunk_rows, rows)
            chunk = tensor[i:end_i]
            result[i:end_i] = torch.where(chunk > 0, 1, -1)
            
            if (i // chunk_rows) % 10 == 0 and torch.cuda.is_available():
                self._clear_memory()
        
        return result

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        enc, indices, is_wrong_left = self.encode_chunked(x, mask, PERCENTAGE, is_wrong)
        
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        if enc.dtype != self.classify_weights.dtype:
            self.classify_weights.data = self.classify_weights.data.to(enc.dtype)

        enc_normalized = self._normalize_chunked(enc, chunk_size=4096)
        logits = F.linear(enc_normalized, self.classify.weight)

        return logits, enc_normalized, indices, is_wrong_left # enc is still hd_dim, but some elements are 0
    
    def _normalize_chunked(self, enc, chunk_size=4096):
        num_samples = enc.shape[0]

        for i in range(0, num_samples, chunk_size):
            end_i = min(i + chunk_size, num_samples)
            chunk = enc[i:end_i]
            norm = torch.norm(chunk, dim=1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            enc[i:end_i] = chunk / norm
        
        return enc

    def get_predictions(self, enc, chunk_size=4096):
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        if enc.dtype != self.classify_weights.dtype:
            self.classify_weights.data = self.classify_weights.data.to(enc.dtype)
        
        num_samples = enc.shape[0]
        all_logits = []
        
        for i in range(0, num_samples, chunk_size):
            end_i = min(i + chunk_size, num_samples)
            enc_chunk = enc[i:end_i]
            logits_chunk = self.classify(F.normalize(enc_chunk))
            all_logits.append(logits_chunk)
            
            del enc_chunk, logits_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logits = torch.cat(all_logits, dim=0)
        return logits
    
    def get_accuracy(self, x, labels):
        """
        Returns accuracy, confidence_map, class_accuracies
        """
        self.eval()
        
        with torch.no_grad():
            enc, _, _ = self.encode_chunked(x)
            
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
    
    def diagnose_hdc_layer(self, dataloader, samples=20000):
        self.eval()
        device = self.device
        D = self.hd_dim
        C = self.num_classes

        bit_sum = torch.zeros(D, device=device)
        count = 0

        pre_sign_var = []
        post_sign_var = []

        proto_sum = torch.zeros(C, D, device=device)
        proto_count = torch.zeros(C, device=device)

        with torch.no_grad():
            for x, _, labels, *_ in dataloader:
                x = x.to(device)
                labels = labels.flatten().to(device)

                enc, _, _ = self.encode_chunked(x)
                signed = torch.sign(enc)

                take = min(enc.size(0), samples - count)

                bit_sum += signed[:take].sum(dim=0)
                count += take

                pre_sign_var.append(enc[:take].var(dim=0).mean())
                post_sign_var.append(signed[:take].var(dim=0).mean())

                for i in range(take):
                    c = labels[i].item()
                    if c == 255:
                        continue
                    proto_sum[c] += signed[i]
                    proto_count[c] += 1

                if count >= samples:
                    break

        bit_imbalance = (bit_sum / count).abs().mean().item()

        pre_var = torch.stack(pre_sign_var).mean().item()
        post_var = torch.stack(post_sign_var).mean().item()
        variance_ratio = post_var / (pre_var + 1e-8)

        proto = torch.sign(proto_sum / proto_count.unsqueeze(1))
        sim = proto @ proto.T / D
        off_diag = sim[~torch.eye(C, dtype=bool, device=device)]
        mean_proto_sim = off_diag.mean().item()

        bit_usage = (proto.abs().mean(dim=0) > 0.1).float().mean().item()

        print("\n===== HDC LAYER DIAGNOSIS =====")
        print(f"Bit imbalance        : {bit_imbalance:.4f}")
        print(f"Variance ratio       : {variance_ratio:.4f}")
        print(f"Mean proto similarity: {mean_proto_sim:.4f}")
        print(f"Effective dims used  : {bit_usage*100:.1f}%")

        return {
            "bit_imbalance": bit_imbalance,
            "variance_ratio": variance_ratio,
            "mean_proto_similarity": mean_proto_sim,
            "effective_dim_fraction": bit_usage,
        }


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
    
    def init_subclusters(self, dataloader, bandwidth=None, chunk_size=2000, max_samples_per_class=5000):
        """
        Initialize subclusters
        """
        self.eval()
        num_sub_per_cluster = self.num_subclusters

        print(f"Collecting embeddings for {self.num_classes} classes")

        all_subcluster_centers = []
        all_subcluster_classes = []
        
        for class_id in range(self.num_classes):
            print(f"Processing class {class_id}...")

            class_embeddings = []
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, npoints) in enumerate(dataloader):
                    proj_in = proj_in.to(self.device)
                    proj_labels = proj_labels.to(self.device).flatten()

                    enc, _, _ = self.encode_chunked(proj_in)

                    class_mask = proj_labels == class_id
                    if torch.any(class_mask):
                        class_enc = enc[class_mask].cpu().half()
                        class_embeddings.append(class_enc)
                        total_samples += class_enc.shape[0]

                    del proj_in, proj_labels
                    self._clear_memory()
                    
                    print(f"  Batch {batch_idx}: collected {total_samples} samples so far")
                    
                    if total_samples >= max_samples_per_class:
                        break

            if not class_embeddings:
                print(f"  No data for class {class_id}, skipping")
                continue

            class_emb_cpu = torch.cat(class_embeddings, dim=0)

            if len(class_emb_cpu) > max_samples_per_class:
                indices = torch.randperm(len(class_emb_cpu))[:max_samples_per_class]
                class_emb_cpu = class_emb_cpu[indices]
            
            class_emb_np = class_emb_cpu.numpy()

            if bandwidth is None:
                estimated_bandwidth = estimate_bandwidth_binary(
                    class_emb_np, 
                    quantile=0.2,
                    n_samples=min(500, len(class_emb_np))
                )
                print(f"  Using Estimated bandwidth for class {class_id}: {estimated_bandwidth:.4f}")
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

    def _process_single_class(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
        """Process a single class to generate its subclusters."""
        if len(class_emb_np) == 0:
            return []
        
        print(f"  Running mean shift on {len(class_emb_np)} samples...")
        cluster_centers = mean_shift_binary(
            X=class_emb_np,
            bandwidth=bandwidth,
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
        print(f"Loading {total_centers} subclusters into model...")

        with torch.no_grad():
            batch_size = 100
            for i in range(0, total_centers, batch_size):
                end_idx = min(i + batch_size, total_centers)

                batch = torch.stack([
                    self._make_bipolar(c.to(self.device)) if c.device.type == 'cpu' 
                    else self._make_bipolar(c) 
                    for c in centers_list[i:end_idx]
                ])
                
                assert torch.all(torch.abs(batch) == 1), f"Subclusters must be bipolar! Got values: {torch.unique(batch)}"

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
        distance_sensitivity introduces a power law which scales with similarity = base_similarity ** (1 / distance_sensitivity)
        """
        enc_binary = torch.sign(enc)
        
        mask = self.subcluster_to_class == class_id
        relevant_subclusters = self.subclusters[mask]
        
        if len(relevant_subclusters) == 0:
            return torch.zeros(enc.shape[0], device=enc.device), None

        hd_dim = enc_binary.shape[1]

        dot_products = torch.matmul(enc_binary, relevant_subclusters.T)

        base_similarity = (dot_products + hd_dim) / (2 * hd_dim)

        if distance_sensitivity == 0.0:
            scaled_similarity = torch.where(
                base_similarity > 0.5,
                torch.tensor(1.0, device=enc.device),
                base_similarity * 2.0
            )
        elif distance_sensitivity == 1.0:
            scaled_similarity = base_similarity
        else:
            exponent = 1.0 / max(distance_sensitivity, 0.001)
            scaled_similarity = base_similarity ** exponent

        max_similarities, relative_indices = torch.max(scaled_similarity, dim=1)
        absolute_indices = torch.nonzero(mask)[relative_indices, 0]
        
        return max_similarities, absolute_indices

    def update(self, x):
        """x being a single image datapoint"""
        enc, _, _ = self.encode(x)
        num_hv = enc.size(0)
        for _, hv in enumerate(enc):
            hv = hv.unsqueeze(0)
            self.update_hv(hv)

    def update_hv(self, x):
        """x being a single hypervector already processed by the model"""
        pred = self.get_predictions(x)
        pred_id = torch.argmax(pred)
        current_prototype = self.classify_weights[pred_id:pred_id+1]

        subcluster_sims, _ = self.get_max_subcluster_similarity(x, pred_id)

        disagreements = (current_prototype * x) == -1
        num_disagreements = disagreements.sum().item()
        if num_disagreements == 0: return

        num_bits_to_flip = int(subcluster_sims[0].item() * num_disagreements)

        disagree_indices = torch.nonzero(disagreements, as_tuple=True)[1]

        flip_indices = disagree_indices[torch.randperm(len(disagree_indices), device=self.device)[:num_bits_to_flip]]

        new_prototype = current_prototype.clone()
        new_prototype[:, flip_indices] *= -1
        
        self.classify_weights[pred_id] = new_prototype.squeeze(0)

    def inference_update(self, x, beta=0.5):
        """
        Inference with updates based on distance.
        If beta=0, then the confidence updates are removed (ablation)
        """
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            enc_normalized = F.normalize(enc)

            if enc_normalized.dtype != self.classify.weight.dtype:
                enc_normalized = enc_normalized.to(self.classify.weight.dtype)
            
            logits = self.classify(enc_normalized)
            predictions = torch.argmax(logits, dim=1)

            enc_binary = torch.sign(enc) # distance calculation
            # prototypes = torch.sign(self.classify.weight)
            # selected_prototypes = prototypes[predictions]
            # hd_dim = enc_binary.shape[1]
            # similarities = torch.sum(enc_binary * selected_prototypes, dim=1) / hd_dim
            # distances = (1 - similarities) / 2

            confidence = torch.softmax(logits, dim=1).max(dim=1)[0] # confidence based mask
            mask = confidence < (1 - beta)

            # top2_logits = torch.topk(logits, 2, dim=1)[0] # relative distances
            # margin = top2_logits[:, 0] - top2_logits[:, 1]
            # mask = margin < beta  # Update when margin is small

            print(f"Distances have mean {confidence.mean()} and variance {confidence.var()}")

            if torch.any(mask):
                distant_hvs = enc_binary[mask]
                distant_predictions = predictions[mask]
                
                unique_classes = torch.unique(distant_predictions)
                
                for class_id in unique_classes:
                    class_mask = distant_predictions == class_id
                    class_hvs = distant_hvs[class_mask]

                    subcluster_sims, _ = self.get_max_subcluster_similarity(class_hvs, class_id)
                    
                    current_prototype = self.classify_weights[class_id:class_id+1]

                    disagreements = (current_prototype * class_hvs) == -1

                    # flip_prob[i, j] = (sum of similarities for samples that disagree at bit j) / N_class
                    flip_weights = disagreements.float() * subcluster_sims.unsqueeze(1)
                    flip_contribution = flip_weights.sum(dim=0)

                    has_disagreement = disagreements.any(dim=0)

                    # each bit gets weighted by how much it should flip
                    update_direction = -2 * current_prototype[0] * has_disagreement.float()
                    weighted_update = update_direction * flip_contribution / (class_hvs.shape[0] + 1e-8)
                    
                    # Apply threshold to actually flip bits (if weighted_update magnitude > 0.5)
                    should_flip = torch.abs(weighted_update) > 0.5
                    
                    self.classify_weights[class_id, should_flip] *= -1
            
            return predictions
        
class NewModel(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, max_subclusters = 5):
        super(NewModel, self).__init__()

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
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)

        self.num_subclusters = max_subclusters
        self.subclusters = nn.Parameter(torch.zeros(self.num_classes * self.num_subclusters, self.hd_dim, device=self.device))
        self.subclusters.data.fill_(0.0)

        self.subcluster_to_class = torch.repeat_interleave(torch.arange(self.num_classes, device=self.device), self.num_subclusters)

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.cuda.amp.autocast(enabled=True):
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
    
    def init_subclusters(self, dataloader, bandwidth=None, max_samples_per_class=5000):
        """
        Initialize subclusters
        """
        self.eval()
        num_sub_per_cluster = self.num_subclusters

        print(f"Collecting embeddings for {self.num_classes} classes")

        all_subcluster_centers = []
        all_subcluster_classes = []
        
        for class_id in range(self.num_classes):
            print(f"Processing class {class_id}...")

            class_embeddings = []
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
                        total_samples += class_enc.shape[0]

                    del proj_in, proj_labels
                    self._clear_memory()
                    
                    print(f"  Batch {batch_idx}: collected {total_samples} samples so far")
                    
                    if total_samples >= max_samples_per_class:
                        break

            if not class_embeddings:
                print(f"  No data for class {class_id}, skipping")
                continue

            class_emb_cpu = torch.cat(class_embeddings, dim=0)

            if len(class_emb_cpu) > max_samples_per_class:
                indices = torch.randperm(len(class_emb_cpu))[:max_samples_per_class]
                class_emb_cpu = class_emb_cpu[indices]
            
            class_emb_np = class_emb_cpu.numpy()

            if bandwidth is None:
                estimated_bandwidth = estimate_bandwidth_binary(
                    class_emb_np, 
                    quantile=0.2,
                    n_samples=min(500, len(class_emb_np))
                )
                print(f"  Using Estimated bandwidth for class {class_id}: {estimated_bandwidth:.4f}")
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

    def _process_single_class(self, class_emb_np, class_id, num_sub_per_cluster, bandwidth):
        """Process a single class to generate its subclusters."""
        if len(class_emb_np) == 0:
            return []
        
        print(f"  Running mean shift on {len(class_emb_np)} samples...")
        cluster_centers = mean_shift_binary(
            X=class_emb_np,
            bandwidth=bandwidth,
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
        print(f"Loading {total_centers} subclusters into model...")

        with torch.no_grad():
            batch_size = 100
            for i in range(0, total_centers, batch_size):
                end_idx = min(i + batch_size, total_centers)

                batch = torch.stack([
                    self._make_bipolar(c.to(self.device)) if c.device.type == 'cpu' 
                    else self._make_bipolar(c) 
                    for c in centers_list[i:end_idx]
                ])
                
                assert torch.all(torch.abs(batch) == 1), f"Subclusters must be bipolar! Got values: {torch.unique(batch)}"

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
        distance_sensitivity introduces a power law which scales with similarity = base_similarity ** (1 / distance_sensitivity)
        """
        enc_binary = torch.sign(enc).to(dtype=self.subclusters.dtype)
        
        mask = self.subcluster_to_class == class_id
        relevant_subclusters = self.subclusters[mask]
        
        if len(relevant_subclusters) == 0:
            return torch.zeros(enc.shape[0], device=enc.device), None

        hd_dim = enc_binary.shape[1]

        dot_products = torch.matmul(enc_binary, relevant_subclusters.T)

        base_similarity = (dot_products + hd_dim) / (2 * hd_dim)

        if distance_sensitivity == 0.0:
            scaled_similarity = torch.where(
                base_similarity > 0.5,
                torch.tensor(1.0, device=enc.device),
                base_similarity * 2.0
            )
        elif distance_sensitivity == 1.0:
            scaled_similarity = base_similarity
        else:
            exponent = 1.0 / max(distance_sensitivity, 0.001)
            scaled_similarity = base_similarity ** exponent

        max_similarities, relative_indices = torch.max(scaled_similarity, dim=1)
        absolute_indices = torch.nonzero(mask)[relative_indices, 0]
        
        return max_similarities, absolute_indices

    def inference_update(self, x, beta=0.5):
        """
        Inference with updates based on distance.
        If beta=0, then the confidence updates are removed (ablation)
        """
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            enc_normalized = F.normalize(enc)

            if enc_normalized.dtype != self.classify.weight.dtype:
                enc_normalized = enc_normalized.to(self.classify.weight.dtype)
            
            logits = self.classify(enc_normalized)
            predictions = torch.argmax(logits, dim=1)

            enc_binary = torch.sign(enc) # distance calculation
            # prototypes = torch.sign(self.classify.weight)
            # selected_prototypes = prototypes[predictions]
            # hd_dim = enc_binary.shape[1]
            # similarities = torch.sum(enc_binary * selected_prototypes, dim=1) / hd_dim
            # distances = (1 - similarities) / 2

            confidence = torch.softmax(logits, dim=1).max(dim=1)[0] # confidence based mask
            mask = confidence < (1 - beta)

            # top2_logits = torch.topk(logits, 2, dim=1)[0] # relative distances
            # margin = top2_logits[:, 0] - top2_logits[:, 1]
            # mask = margin < beta  # Update when margin is small

            if torch.any(mask):
                distant_hvs = enc_binary[mask]
                distant_predictions = predictions[mask]
                
                unique_classes = torch.unique(distant_predictions)
                
                for class_id in unique_classes:
                    class_mask = distant_predictions == class_id
                    class_hvs = distant_hvs[class_mask]

                    subcluster_sims, _ = self.get_max_subcluster_similarity(class_hvs, class_id)
                    
                    current_prototype = self.classify_weights[class_id:class_id+1]

                    disagreements = (current_prototype * class_hvs) == -1

                    # flip_prob[i, j] = (sum of similarities for samples that disagree at bit j) / N_class
                    flip_weights = disagreements.float() * subcluster_sims.unsqueeze(1)
                    flip_contribution = flip_weights.sum(dim=0)

                    has_disagreement = disagreements.any(dim=0)

                    # each bit gets weighted by how much it should flip
                    update_direction = -2 * current_prototype[0] * has_disagreement.float()
                    weighted_update = update_direction * flip_contribution / (class_hvs.shape[0] + 1e-8)
                    
                    # Apply threshold to actually flip bits (if weighted_update magnitude > 0.5)
                    should_flip = torch.abs(weighted_update) > 0.5
                    
                    self.classify_weights[class_id, should_flip] *= -1
            
            return predictions

def set_new_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
    return NewModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)
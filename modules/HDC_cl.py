from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from tqdm import tqdm

# torchhd for HD encoding  (same dependency as the original)
import torchhd.embeddings as embeddings
import torchhd.functional as functional

from modules.HDC_utils import mean_shift_binary, estimate_bandwidth_binary

class _ResBlock2D(nn.Module):
    """Tiny 2-D residual block used inside both encoders."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))

class PointPillarEncoder(nn.Module):
    """
    PointPillars-style encoder → (B, 128) scene embedding.

    Input  : voxelised point cloud as a dict with keys
               'voxel_features'  (N_voxels, max_pts, C)
               'voxel_coords'    (N_voxels, 4)   [batch, z, y, x]
               'batch_size'      int
             OR a range image tensor (B, C, H, W) passed directly.

    Output : (B, 128)  – one embedding vector per sample in the batch.

    The spatial BEV feature map (B, 128, H, W) is reduced to (B, 128) via
    adaptive average pooling so that DensityModel.encode() receives exactly
    the same channel width (128) it expects, but as a 1-D embedding rather
    than a spatial map.
    """

    def __init__(
        self,
        in_channels: int = 4,
        bev_shape: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_shape

        self.vfn = nn.Sequential(
            nn.Linear(in_channels, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.bev_backbone = nn.Sequential(
            _ResBlock2D(128, 128),
            _ResBlock2D(128, 256),
            _ResBlock2D(256, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch keys
        ----------
        voxel_features : (N, max_pts, C)
        voxel_coords   : (N, 4)  [batch_idx, z, y, x]
        batch_size     : int
        """
        vf = batch["voxel_features"]
        vc = batch["voxel_coords"]
        B  = int(batch["batch_size"])

        vf = vf.max(dim=1).values
        vf = self.vfn(vf)

        # Scatter to BEV canvas
        bev = vf.new_zeros(B, 128, self.bev_h, self.bev_w)
        b_idx = vc[:, 0].long()
        y_idx = vc[:, 2].long().clamp(0, self.bev_h - 1)
        x_idx = vc[:, 3].long().clamp(0, self.bev_w - 1)
        bev[b_idx, :, y_idx, x_idx] = vf

        feat = self.bev_backbone(bev)
        feat = self.pool(feat).flatten(1)
        return feat

    def forward_range_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alternative path for datasets that supply a range image (B, C, H, W)
        rather than raw voxels (e.g. V2X-R projected LiDAR).

        Returns (B, 128).
        """
        B, C, H, W = x.shape
        if not hasattr(self, "_ri_stem") or self._ri_stem[0].in_channels != C:
            self._ri_stem = nn.Sequential(
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ).to(x.device)
        x = self._ri_stem(x)                   # (B, 128, H, W)
        x = self.bev_backbone(x)               # (B, 128, H', W')
        return self.pool(x).flatten(1)         # (B, 128)

class RadarTensorEncoder(nn.Module):
    """
    Encoder for K-Radar 4-D radar power spectra.

    K-Radar provides a (Range x Azimuth x Elevation x Doppler) tensor.
    We first collapse elevation and Doppler via max-pooling to produce a
    (B, 1, R, A) range-azimuth map, then run a lightweight 2-D CNN.

    Input  : (B, 1, R, A, E, D)  or pre-collapsed (B, 1, R, A)
    Output : (B, 128)
    """

    def __init__(
        self,
        range_bins: int = 256,
        azimuth_bins: int = 107,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            _ResBlock2D(32, 64),
            nn.MaxPool2d(2),
            _ResBlock2D(64, 128),
            nn.MaxPool2d(2),
            _ResBlock2D(128, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, R, A)        – range-azimuth power map (already collapsed)
            OR
            (B, 1, R, A, E, D)  – full 4-D tensor (collapsed here)
        Returns : (B, 128)
        """
        if x.dim() == 6:
            x = x.amax(dim=5).amax(dim=4)

        feat = self.backbone(x)
        return self.pool(feat).flatten(1)

class ClassificationDensityModel(nn.Module):
    """
    HDC density model adapted for **per-sample classification**.

    Differences from the original DensityModel
    -------------------------------------------
    encode()
        The external backbone (PointPillarEncoder or RadarTensorEncoder)
        already returns (B, 128).  We do NOT reshape/permute spatial dims.
        Returns: (sample_hv,)   shape (B, hd_dim)

    forward()
        Returns (logits, enc_norm) — shapes (B, num_classes), (B, hd_dim).
        No `indices` or `is_wrong_left`; those existed only to support the
        partial-sample retraining trick in the segmentation pipeline.

    get_accuracy()
        Plain top-1 accuracy over the batch.  No spatial reshape.

    inference_update()
        Background-pixel mask removed (valid_enc_mask).  Every sample in the
        batch is a real object/scene, so nothing needs to be ignored.
        All other logic — EMA pull, subcluster gating, momentum, FPS
        subsampling — is identical to the original.

    Everything else is byte-for-byte identical to the original DensityModel:
        HD encoding (rp / idlevel / nonlinear)
        hard_quantize
        classify weight layout  (num_classes × hd_dim)
        subcluster parameters and subcluster_to_class
        proto_momentum buffer
        init_subclusters / update_subclusters / _process_single_class
        get_max_subcluster_similarity
        _make_bipolar / _clear_memory / _farthest_point_sample / _stratified_sample
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        hd_encoder: str = "rp",
        num_levels: int = 0,
        randomness: float = 0.0,
        max_subclusters: int = 10,
        subcluster_type: str = "bipolar",
        gauss_rp: bool = True,
    ):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.hd_dim = 10000
        self.temperature = 0.01
        self.input_dim = 128
        self.hd_encoder = hd_encoder
        self.subcluster_type = subcluster_type
        self.num_subclusters = max_subclusters

        self.backbone = backbone

        if self.hd_encoder == "rp":
            torch_rng = torch.get_rng_state()
            numpy_rng = np.random.get_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
            with torch.no_grad():
                if gauss_rp:
                    G = torch.randn(self.hd_dim, self.input_dim)
                    Q, _ = torch.linalg.qr(G)
                    self.projection.weight.copy_(Q * torch.sqrt(torch.tensor(float(self.hd_dim))))
                else:
                    G = torch.randn(self.hd_dim, self.input_dim)
                    self.projection.weight.copy_(G / np.sqrt(self.input_dim))

            torch.set_rng_state(torch_rng)
            np.random.set_state(numpy_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
        elif self.hd_encoder == "idlevel":
            self.value = embeddings.Level(num_levels, self.hd_dim, randomness=randomness)
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
        elif self.hd_encoder == "nonlinear":
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)

        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(device)

        self.subclusters = nn.Parameter(
            torch.zeros(
                self.num_classes * self.num_subclusters,
                self.hd_dim,
                device=self.device,
            )
        )
        self.subclusters.data.fill_(0.0)

        self.subcluster_to_class = torch.repeat_interleave(
            torch.arange(self.num_classes, device=self.device),
            self.num_subclusters,
        )

        # Mean-shift hyper-params (identical defaults)
        self.quantile = 0.4
        self.mult     = 0.2
        self.dedup    = 0.7

        self.gauss_rp = gauss_rp

        # EMA momentum buffer  (identical)
        self.register_buffer(
            "proto_momentum",
            torch.zeros_like(self.classify.weight.data),
        )

    # ======================================================================
    # encode  –  ONE HV PER SAMPLE  (key difference from original)
    # ======================================================================

    def encode(self, x, backbone_kwargs: Optional[dict] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor or dict
            Either a raw tensor forwarded straight to the backbone, or a dict
            (used by PointPillarEncoder which needs voxel_coords etc.).
        backbone_kwargs : dict, optional
            Extra keyword arguments forwarded to backbone.forward().

        Returns
        -------
        sample_hv : (B, hd_dim)
            One hypervector per sample.  Already hard-quantised.
        """
        backbone_kwargs = backbone_kwargs or {}

        with torch.amp.autocast("cuda", enabled=True):
            if isinstance(x, dict):
                feat = self.backbone(x, **backbone_kwargs)   # (B, 128)
            else:
                feat = self.backbone(x, **backbone_kwargs)   # (B, 128)

        # feat : (B, 128)  — no spatial dims to flatten
        B = feat.shape[0]
        mask = torch.ones(self.hd_dim, device=self.device, dtype=torch.bool)

        sample_hv = torch.zeros(B, self.hd_dim, device=self.device, dtype=feat.dtype)

        if self.hd_encoder == "rp":
            if feat.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(feat.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(feat)[:, mask]

        elif self.hd_encoder == "idlevel":
            tmp_hv = functional.bind(
                self.position.weight[:, mask],
                self.value(feat)[:, :, mask],
            )
            sample_hv[:, mask] = functional.multiset(tmp_hv)

        elif self.hd_encoder == "nonlinear":
            sample_hv[:, mask] = self.nonlinear_projection(feat)[:, mask]

        else:
            return feat   # identity path

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv   # (B, hd_dim)

    # ======================================================================
    # forward
    # ======================================================================

    def forward(self, x, backbone_kwargs: Optional[dict] = None):
        """
        Returns
        -------
        logits   : (B, num_classes)
        enc_norm : (B, hd_dim)   — L2-normalised hypervectors
        """
        enc = self.encode(x, backbone_kwargs)          # (B, hd_dim)

        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)

        enc_norm = F.normalize(enc)
        logits   = self.classify(enc_norm)             # (B, num_classes)
        return logits, enc_norm

    # ======================================================================
    # get_predictions  (unchanged)
    # ======================================================================

    def get_predictions(self, enc: torch.Tensor) -> torch.Tensor:
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        return self.classify(F.normalize(enc))

    # ======================================================================
    # get_accuracy  –  per-sample top-1, no spatial reshape
    # ======================================================================

    def get_accuracy(self, x, labels: torch.Tensor, backbone_kwargs: Optional[dict] = None):
        """
        Parameters
        ----------
        x      : input to encode()
        labels : (B,) integer class indices

        Returns
        -------
        accuracy   : float
        per_class  : dict {class_id: float}
        """
        self.eval()
        with torch.no_grad():
            enc    = self.encode(x, backbone_kwargs)   # (B, hd_dim)
            logits = self.get_predictions(enc)         # (B, num_classes)
            preds  = logits.argmax(dim=1)              # (B,)

        labels = labels.to(preds.device).flatten()
        correct = (preds == labels).sum().item()
        total   = labels.numel()
        accuracy = correct / total

        per_class: Dict[int, float] = {}
        for cls in labels.unique():
            c = cls.item()
            if c == 255:
                continue
            mask      = labels == cls
            cls_correct = (preds[mask] == cls).sum().item()
            per_class[c] = cls_correct / mask.sum().item()

        return accuracy, per_class

    # ======================================================================
    # inference_update  –  identical logic, background mask removed
    # ======================================================================

    def inference_update(
        self,
        x,
        beta: float = 0.2,
        distance_sensitivity: float = 1.0,
        learning_rate: float = 0.01,
        chunk_size: int = -1,
        max_updates_per_class: int = -1,
        thresholds: List[float] = None,
        backbone_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Unsupervised prototype update via EMA pull toward confident samples.

        Identical to the original inference_update except:
          - The background-pixel validity mask
              valid_enc_mask = torch.any(enc != 0, dim=1)
            is removed.  Every row of enc is a real sample, not a padded pixel.
          - Returns predictions shaped (B,) instead of (num_pixels,).

        All other logic — distance threshold, subcluster gating, FPS
        subsampling, momentum EMA — is unchanged.
        """
        if thresholds is None:
            thresholds = [0.45, 0.80]

        self.train()
        with torch.no_grad():
            enc = self.encode(x, backbone_kwargs)      # (B, hd_dim)
            B   = enc.shape[0]

            enc_norm = F.normalize(enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)

            curr_chunk = B if chunk_size == -1 else chunk_size

            all_preds        = []
            all_update_masks = []

            if self.subcluster_type == "bipolar":
                proto_binary = torch.sign(self.classify.weight)

            for i in range(0, B, curr_chunk):
                chunk_norm   = enc_norm[i : i + curr_chunk]
                chunk_logits = self.classify(chunk_norm)
                chunk_preds  = chunk_logits.argmax(dim=1)
                all_preds.append(chunk_preds)

                if self.subcluster_type == "bipolar":
                    chunk_enc_orig = enc[i : i + curr_chunk]
                    enc_binary     = torch.sign(chunk_enc_orig)
                    selected_proto = proto_binary[chunk_preds]
                    sims = torch.sum(enc_binary * selected_proto, dim=1) / self.hd_dim
                else:
                    selected_proto = F.normalize(self.classify.weight[chunk_preds])
                    sims = torch.sum(chunk_norm * selected_proto, dim=1)

                distances = (1.0 - sims) / 2.0
                all_update_masks.append(distances > beta)

            predictions = torch.cat(all_preds)           # (B,)
            update_mask = torch.cat(all_update_masks)    # (B,)

            if not torch.any(update_mask):
                return predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(predictions[valid_indices])

            for class_id in unique_classes:
                c_id = class_id.item()

                class_mask    = (predictions == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_idx       = self._farthest_point_sample(
                        enc_norm[class_indices].cpu(), max_updates_per_class
                    )
                    class_indices = class_indices[fps_idx.to(self.device)]

                sample_encs = enc_norm[class_indices]   # (K, hd_dim)

                if self.subcluster_type == "bipolar":
                    target_encs = torch.sign(enc[class_indices])
                    sub_sims, _ = self.get_max_subcluster_similarity(
                        target_encs, c_id, distance_sensitivity
                    )
                else:
                    sub_sims, _ = self.get_max_subcluster_similarity(
                        sample_encs, c_id, distance_sensitivity
                    )

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims    = sub_sims[valid_mask]

                weights            = sub_sims / sub_sims.sum()
                weighted_pull      = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr       = learning_rate * sub_sims.mean().item()

                current_weight     = self.classify.weight[c_id]
                self.proto_momentum[c_id] = (
                    0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull
                )
                updated_weight = (
                    (1.0 - effective_lr) * current_weight
                    + effective_lr * self.proto_momentum[c_id]
                )
                self.classify.weight[c_id] = F.normalize(
                    updated_weight.unsqueeze(0), dim=1
                ).squeeze(0)

        return predictions

    # ======================================================================
    # Subcluster API  –  identical to original
    # ======================================================================

    def get_max_subcluster_similarity(
        self,
        enc: torch.Tensor,
        class_id: int,
        distance_sensitivity: float = 1.0,
    ):
        mask                = self.subcluster_to_class == class_id
        relevant_subclusters = self.subclusters[mask].float()

        if self.subcluster_type == "bipolar":
            enc_binary   = torch.sign(enc).float()
            hd_dim       = enc_binary.shape[1]
            dot_products = torch.matmul(enc_binary, relevant_subclusters.T)
            base_sim     = (dot_products + hd_dim) / (2 * hd_dim)
        elif self.subcluster_type == "continuous":
            enc_norm  = F.normalize(enc.float(), dim=1)
            sub_norm  = F.normalize(relevant_subclusters, dim=1)
            cos_sim   = torch.matmul(enc_norm, sub_norm.T)
            base_sim  = (cos_sim + 1) / 2
        else:
            raise ValueError(f"Unknown subcluster_type: {self.subcluster_type}")

        if distance_sensitivity == 0.0:
            scaled = torch.where(
                base_sim > 0.5,
                torch.tensor(1.0, device=enc.device),
                base_sim * 2.0,
            )
        elif distance_sensitivity == 1.0:
            scaled = base_sim
        else:
            scaled = base_sim ** distance_sensitivity

        max_sims, rel_idx = torch.max(scaled, dim=1)
        abs_idx           = torch.nonzero(mask)[rel_idx, 0]
        return max_sims, abs_idx

    # ------------------------------------------------------------------

    def init_subclusters(
        self,
        dataloader,
        encode_fn=None,
        bandwidth=None,
        max_samples_per_class: int = 8000,
        sampling_strategy: str = "diverse",
    ):
        """
        Identical algorithm to original init_subclusters.

        Parameters
        ----------
        dataloader  : yields (inputs, labels) tuples where labels are (B,)
                      integer class indices  [not flattened pixel maps].
        encode_fn   : optional callable (batch) -> torch.Tensor of shape (B, 128)
                      if the batch format needs custom unpacking before encode().
                      Defaults to self.encode(batch).
        """
        self.eval()
        print(
            f"Collecting embeddings for {self.num_classes} classes "
            f"using '{sampling_strategy}' sampling"
        )
        all_centers = []
        all_classes = []
        MAX_SAMPLES = max_samples_per_class * 2

        for class_id in range(self.num_classes):
            print(f"Processing class {class_id}...")
            class_embs   = []
            batch_tags   = []
            total        = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    inputs, labels = batch[0], batch[1]

                    if encode_fn is not None:
                        enc = encode_fn(inputs)
                    else:
                        if not isinstance(inputs, torch.Tensor):
                            inputs = inputs.to(self.device)
                        enc = self.encode(inputs)          # (B, hd_dim)

                    labels = labels.to(self.device).flatten()
                    cmask  = labels == class_id

                    if torch.any(cmask):
                        cls_enc = enc[cmask].cpu().half()
                        class_embs.append(cls_enc)
                        batch_tags.extend([batch_idx] * cls_enc.shape[0])
                        total += cls_enc.shape[0]

                    del inputs, labels
                    self._clear_memory()

                    if total >= MAX_SAMPLES:
                        break

            if not class_embs:
                print(f"  No data for class {class_id}, skipping")
                continue

            class_emb_cpu = torch.cat(class_embs, dim=0)
            batch_indices = torch.as_tensor(batch_tags[: len(class_emb_cpu)])

            if len(class_emb_cpu) > MAX_SAMPLES:
                idx           = torch.randperm(len(class_emb_cpu))[:MAX_SAMPLES]
                class_emb_cpu = class_emb_cpu[idx]
                batch_indices = batch_indices[idx]

            if len(class_emb_cpu) > max_samples_per_class:
                if sampling_strategy == "random":
                    idx = torch.randperm(len(class_emb_cpu))[:max_samples_per_class]
                elif sampling_strategy == "diverse":
                    idx = self._stratified_sample(batch_indices, max_samples_per_class)
                elif sampling_strategy == "fps":
                    idx = self._farthest_point_sample(class_emb_cpu, max_samples_per_class)
                else:
                    raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
                class_emb_cpu = class_emb_cpu[idx]
                print(
                    f"  Sampled {len(class_emb_cpu)} from {len(batch_tags)} "
                    f"using '{sampling_strategy}'"
                )

            class_emb_np = class_emb_cpu.numpy()

            if bandwidth is None:
                bw = estimate_bandwidth_binary(
                    class_emb_np,
                    quantile=self.quantile,
                    n_samples=min(500, len(class_emb_np)),
                    bandwidth_multiplier=self.mult,
                )
                print(f"  Estimated bandwidth: {bw:.4f}")
            else:
                bw = bandwidth

            del class_emb_cpu, class_embs
            self._clear_memory()

            centers = self._process_single_class(
                class_emb_np, class_id, self.num_subclusters, bw
            )
            all_centers.extend(centers)
            all_classes.extend([class_id] * len(centers))

            del class_emb_np
            self._clear_memory()

        self._load_subclusters(all_centers, all_classes)
        print("Subcluster initialisation complete.")

    # ------------------------------------------------------------------

    def update_subclusters(
        self,
        enc: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float = 0.1,
        min_samples: int = 10,
        method: str = "proximity_pull",
    ):
        """
        Update subclusters from labelled embeddings.

        Parameters
        ----------
        enc    : (B, hd_dim)  – already encoded hypervectors
        labels : (B,)         – integer class labels

        Everything else is identical to original update_subclusters.
        """
        self.eval()
        with torch.no_grad():
            labels_flat = labels.view(-1)

            for class_id in range(self.num_classes):
                class_mask = labels_flat == class_id
                if class_mask.sum() < min_samples:
                    continue

                class_enc          = enc[class_mask].float()
                sub_mask           = self.subcluster_to_class == class_id
                relevant_subclusters = self.subclusters[sub_mask]
                sub_indices        = torch.nonzero(sub_mask).squeeze(1)
                n_subs             = relevant_subclusters.shape[0]

                if self.subcluster_type == "bipolar":
                    class_enc_bin = torch.sign(class_enc).to(self.subclusters.dtype)
                    sims = (
                        torch.matmul(class_enc_bin, relevant_subclusters.T) + self.hd_dim
                    ) / (2 * self.hd_dim)
                else:
                    sims = torch.matmul(
                        F.normalize(class_enc, dim=1),
                        F.normalize(relevant_subclusters, dim=1).T,
                    )

                if method == "proximity_pull":
                    assignments   = torch.argmax(sims, dim=1)
                    asn_exp       = assignments.unsqueeze(1).expand(-1, class_enc.shape[1])

                    sum_per_sub   = torch.zeros(
                        n_subs, class_enc.shape[1], device=self.device, dtype=torch.float32
                    )
                    sum_per_sub.scatter_add_(0, asn_exp, class_enc)

                    counts        = torch.zeros(n_subs, device=self.device, dtype=torch.float32)
                    counts.scatter_add_(0, assignments, torch.ones(assignments.shape[0], device=self.device))

                    valid = counts >= min_samples
                    if not valid.any():
                        continue
                    new_means = sum_per_sub[valid] / counts[valid].unsqueeze(1)

                elif method == "soft_weighted":
                    weights    = F.softmax(sims, dim=1)
                    new_means  = torch.matmul(weights.T, class_enc)
                    weight_sums = weights.sum(dim=0)
                    valid      = weight_sums >= (min_samples * 0.1)
                    if not valid.any():
                        continue
                    new_means = new_means[valid] / weight_sums[valid].unsqueeze(1)

                elif method == "mean_shift":
                    class_emb_np = class_enc.cpu().numpy()
                    bw = estimate_bandwidth_binary(
                        class_emb_np,
                        quantile=self.quantile,
                        n_samples=min(500, len(class_emb_np)),
                        bandwidth_multiplier=self.mult,
                    )
                    try:
                        new_centers = mean_shift_binary(
                            X=class_emb_np,
                            bandwidth=bw,
                            quantile=self.quantile,
                            bandwidth_multiplier=self.mult,
                            dedup_scale=self.dedup,
                        )
                    except Exception as e:
                        print(f"  Mean shift failed for class {class_id}: {e}")
                        continue

                    new_centers   = np.sign(new_centers)
                    new_centers_t = F.normalize(
                        torch.tensor(new_centers, dtype=torch.float32, device=self.device), dim=1
                    )

                    for nc in new_centers_t:
                        if self.subcluster_type == "bipolar":
                            nc_bin = torch.sign(nc).to(self.subclusters.dtype)
                            s = (
                                torch.matmul(nc_bin.unsqueeze(0), relevant_subclusters.T)
                                + self.hd_dim
                            ) / (2 * self.hd_dim)
                        else:
                            s = torch.matmul(
                                F.normalize(nc.unsqueeze(0), dim=1),
                                F.normalize(relevant_subclusters, dim=1).T,
                            )

                        s         = s.squeeze(0)
                        closest   = s.argmax().item()
                        abs_idx   = sub_indices[closest].item()
                        current   = self.subclusters.data[abs_idx].float()
                        updated   = (1.0 - learning_rate) * current + learning_rate * nc.float()

                        if self.subcluster_type == "bipolar":
                            updated         = torch.sign(updated)
                            updated[updated == 0] = -1.0

                        self.subclusters.data[abs_idx] = F.normalize(
                            updated.unsqueeze(0), dim=1
                        ).squeeze(0)
                        relevant_subclusters = self.subclusters[sub_mask]

                    continue   # skip generic EMA update below

                else:
                    raise ValueError(f"Unknown method: {method}")

                # Generic EMA update (proximity_pull / soft_weighted)
                if self.subcluster_type == "bipolar":
                    new_means         = torch.sign(new_means)
                    new_means[new_means == 0] = -1.0
                new_means = F.normalize(new_means, dim=1)

                valid_abs    = sub_indices[valid]
                current      = self.subclusters.data[valid_abs].float()
                updated      = (1.0 - learning_rate) * current + learning_rate * new_means

                if self.subcluster_type == "bipolar":
                    updated           = torch.sign(updated)
                    updated[updated == 0] = -1.0

                self.subclusters.data[valid_abs] = F.normalize(updated, dim=1)

    # ======================================================================
    # Private helpers  –  all identical to original
    # ======================================================================

    def _stratified_sample(self, batch_indices: torch.Tensor, n_samples: int) -> torch.Tensor:
        unique_batches     = torch.unique(batch_indices)
        samples_per_batch  = n_samples // len(unique_batches)
        remainder          = n_samples % len(unique_batches)
        selected           = []

        for i, bid in enumerate(unique_batches):
            positions  = torch.where(batch_indices == bid)[0]
            n_take     = min(samples_per_batch + (1 if i < remainder else 0), len(positions))
            perm       = torch.randperm(len(positions))[:n_take]
            selected.append(positions[perm])

        return torch.cat(selected)

    def _farthest_point_sample(self, embeddings: torch.Tensor, n_samples: int) -> torch.Tensor:
        n = len(embeddings)
        if n <= n_samples:
            return torch.arange(n)

        selected  = [torch.randint(0, n, (1,)).item()]
        distances = torch.full((n,), float("inf"))

        for _ in range(n_samples - 1):
            last      = embeddings[selected[-1]]
            new_dist  = torch.sum((embeddings - last) ** 2, dim=1)
            distances = torch.minimum(distances, new_dist)
            farthest  = torch.argmax(distances).item()
            selected.append(farthest)
            distances[farthest] = 0

        return torch.tensor(selected)

    def _process_single_class(
        self,
        class_emb_np: np.ndarray,
        class_id: int,
        num_sub: int,
        bandwidth: float,
    ) -> List:
        if len(class_emb_np) == 0:
            return []

        print(f"  Running mean shift on {len(class_emb_np)} samples...")
        centers = mean_shift_binary(
            X=class_emb_np,
            bandwidth=bandwidth,
            quantile=self.quantile,
            bandwidth_multiplier=self.mult,
            dedup_scale=self.dedup,
        )

        if self.subcluster_type == "bipolar":
            centers = np.sign(centers)

        print(f"  Found {len(centers)} clusters")

        subclusters = []
        if len(centers) <= num_sub:
            for c in centers:
                subclusters.append(torch.tensor(c, dtype=torch.float32))
        else:
            ct    = torch.tensor(centers, dtype=torch.float32)
            idx   = self._farthest_point_sample(ct, num_sub)
            for i in idx.tolist():
                subclusters.append(torch.tensor(centers[i], dtype=torch.float32))

        return subclusters

    def _load_subclusters(self, centers_list: List, classes_list: List):
        if not centers_list:
            print("Warning: no subclusters to load.")
            return

        total = len(centers_list)
        print(f"Loading {total} subclusters (type: {self.subcluster_type})...")

        with torch.no_grad():
            batch = 100
            for i in range(0, total, batch):
                end  = min(i + batch, total)
                if self.subcluster_type == "bipolar":
                    chunk = torch.stack(
                        [self._make_bipolar(c.to(self.device)) for c in centers_list[i:end]]
                    )
                elif self.subcluster_type == "continuous":
                    chunk = torch.stack(
                        [c.to(self.device) for c in centers_list[i:end]]
                    )
                    chunk = F.normalize(chunk, dim=1)
                else:
                    raise ValueError(f"Unknown subcluster_type: {self.subcluster_type}")

                self.subclusters.data[i:end] = chunk
                del chunk
                if i % 500 == 0:
                    self._clear_memory()

        print("All subclusters loaded.")

    def _make_bipolar(self, tensor: torch.Tensor) -> torch.Tensor:
        result = torch.sign(tensor)
        result[result == 0] = -1
        return result

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

class DensityClassifier:
    """
    Supervised HDC training + unsupervised inference-update loop for
    object / scene classification.

    Mirrors the structure of DensityTrainer (segmentation) but targets
    per-sample labels instead of per-pixel labels.

    Parameters
    ----------
    model          : ClassificationDensityModel
    num_classes    : int
    device         : torch.device
    loss_weights   : (num_classes,) tensor  –  optional class weights for
                     display / logging  (HDC training itself is weight-free)
    ignore_classes : list[int]  –  classes excluded from IoU / accuracy eval
    epochs         : int  –  number of retraining (refinement) epochs
    bipolar_prototypes : bool  –  binarise prototype weights after each pass
    """

    def __init__(
        self,
        model: ClassificationDensityModel,
        num_classes: int,
        device: torch.device,
        loss_weights: Optional[torch.Tensor] = None,
        ignore_classes: Optional[List[int]] = None,
        epochs: int = 20,
        bipolar_prototypes: bool = False,
    ):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.loss_weights = loss_weights
        self.ignore_classes = ignore_classes or []
        self.epochs = epochs
        self.bipolar_prototypes = bipolar_prototypes

        self.model.to(device)

        if torch.cuda.is_available():
            cudnn.benchmark = True
            cudnn.fastest = True

        self._is_wrong_list: List[Optional[torch.Tensor]] = []

    def start(self, train_loader, val_loader):
        """
        Run one initial training pass then `self.epochs` refinement epochs,
        validating after each refinement epoch.

        train_loader / val_loader yield:
            (inputs, labels, ...)
        where
            inputs : whatever the backbone accepts
            labels : (B,) integer class indices
        """
        print("Starting HDC classification training...")
        self._is_wrong_list = [None] * len(train_loader)

        t0 = time.time()
        self._train_initial(train_loader)
        print(f"Initial train pass done in {time.time() - t0:.1f}s")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self._train_retrain(train_loader, epoch)
            print(f"Retrain epoch {epoch} done in {time.time() - t0:.1f}s")

            acc = self.validate(val_loader)
            print(f"Validation accuracy after epoch {epoch}: {acc:.4f}")

    def _train_initial(self, train_loader):
        """
        Accumulate HVs into classify_weights using index_add_ —
        identical bookkeeping to DensityTrainer.train(), but labels are
        per-sample (B,) not per-pixel (B*H*W,).
        """
        self.model.eval()
        train_times = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader, desc="Initial train")):
                inputs, labels = self._unpack(batch)

                t0 = time.time()
                sample_hv = self.model.encode(inputs)
                sample_hv = sample_hv.to(self.model.classify_weights.dtype)

                labels_flat = labels.view(-1).to(self.device)

                self.model.classify_weights.index_add_(0, labels_flat, sample_hv)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_times.append(time.time() - t0)

                preds = self.model.get_predictions(sample_hv).argmax(dim=1)
                is_wrong = labels_flat != preds
                self._is_wrong_list[i] = is_wrong

            self._apply_prototype_normalisation()

        print(f"  Mean train time: {np.mean(train_times):.4f}s ± {np.std(train_times):.4f}s")
        wrong_total = sum(w.sum().item() for w in self._is_wrong_list if w is not None)
        print(f"  Initial wrong classifications: {wrong_total}")

    def _train_retrain(self, train_loader, epoch: int):
        """
        For each misclassified sample:
          classify_weights[true]  += 2 * hv
          classify_weights[wrong] -= 2 * hv
        """
        self.model.eval()
        total_miss  = 0
        retrain_times = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader, desc=f"Retrain epoch {epoch}")):
                inputs, labels = self._unpack(batch)

                self._apply_prototype_normalisation()

                t0 = time.time()
                logits, sample_hv = self.model(inputs)
                preds = logits.argmax(dim=1)
                labels_flat = labels.view(-1).to(self.device)

                is_wrong = labels_flat != preds
                if is_wrong.sum().item() == 0:
                    self._is_wrong_list[i] = is_wrong
                    retrain_times.append(time.time() - t0)
                    continue

                total_miss += is_wrong.sum().item()

                wrong_labels = labels_flat[is_wrong]
                wrong_preds = preds[is_wrong]
                wrong_hv = sample_hv[is_wrong].to(self.model.classify_weights.dtype)

                self.model.classify_weights.index_add_(0, wrong_labels, wrong_hv)
                self.model.classify_weights.index_add_(0, wrong_preds, -wrong_hv)

                self._is_wrong_list[i] = is_wrong

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                retrain_times.append(time.time() - t0)

        print(f"  total_miss: {total_miss}  |  mean retrain time: {np.mean(retrain_times):.4f}s")

    def validate(self, val_loader) -> float:
        """
        Returns overall top-1 accuracy across the validation set.
        """
        self.model.eval()
        correct = 0
        total = 0
        val_times = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, labels = self._unpack(batch)
                labels_flat = labels.view(-1).to(self.device)

                t0 = time.time()
                logits, _ = self.model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                val_times.append(time.time() - t0)

                preds = logits.argmax(dim=1)
                correct += (preds == labels_flat).sum().item()
                total += labels_flat.numel()

        acc = correct / total if total > 0 else 0.0
        print(f"  Validation — accuracy: {acc:.4f}  ({correct}/{total})  mean time: {np.mean(val_times):.4f}s")
        return acc

    def _apply_prototype_normalisation(self):
        """Normalise or binarise prototype weights — identical to original."""
        with torch.no_grad():
            if self.bipolar_prototypes:
                data = torch.sign(self.model.classify_weights.data)
                zeros = data == 0
                if zeros.any():
                    data[zeros] = -1.0
                self.model.classify_weights.data = data
                self.model.classify.weight.data  = data.clone()
            else:
                self.model.classify.weight[:] = F.normalize(self.model.classify_weights)

    def _unpack(self, batch) -> Tuple[any, torch.Tensor]:
        """
        Unpack a dataloader batch into (inputs, labels).

        Handles three common formats:
          (inputs, labels)
          (inputs, labels, ...)   – extra fields ignored
          dict with keys 'inputs'/'data' and 'labels'/'targets'/'gt'
        """
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            labels = batch[1]
        elif isinstance(batch, dict):
            inputs = batch.get("inputs", batch.get("data"))
            labels = batch.get("labels", batch.get("targets", batch.get("gt")))
        else:
            raise ValueError(f"Unrecognised batch type: {type(batch)}")

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels
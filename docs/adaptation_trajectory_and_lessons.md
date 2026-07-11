# Unsupervised Adaptation: Trajectory & Lessons Learned

## 1. Project Goal & Initial State
The objective is to build a robust, unsupervised Test-Time Adaptation (TTA) pipeline for LiDAR semantic segmentation, capable of adapting a pre-trained feature extractor to severe synthetic corruptions (KITTI-C) and massive cross-sensor domain shifts (KITTI -> NuScenes).

Our primary experimental method was **`exp_a` (Subcluster-Gated Soft Consensus)**. This method relied on two core concepts:
1. **Soft Consensus (TTAug):** Using geometric augmentations (jitter, scale) to gauge prediction stability.
2. **Subcluster Gating:** Filtering out confident noise by measuring a point's similarity to exact training distribution modes (subclusters) rather than just the class average (prototypes).

## 2. The Problem: Catastrophic Collapse
During early testing, the `density` baseline (which uses simple linear prototype updating) remained stable and improved mIoU. In contrast, `exp_a` suffered from catastrophic collapse, where mIoU would plummet to near-zero on specific corruptions (e.g., Fog) almost instantly.

## 3. What We Tested & Why

### Test 1: Augmentation Pipeline Integrity
- **Why:** TTAug relies heavily on producing realistic views. We suspected the augmentation pipeline was destroying features instead of simulating geometry.
- **Findings:** `F.dropout2d` was being used to simulate beam-missing. However, it was dropping out random feature channels across the spatial dimension, completely scrambling the semantic embeddings. 
- **Action:** Replaced dropout with a geometric depth scaling (`x * 0.95`), stabilizing the consensus logic.

### Test 2: Update Magnitude (Volume Weighting)
- **Why:** Even with healthy augmentations, `exp_a` continued to collapse. We suspected the mathematical magnitude of the updates was too aggressive compared to the baseline.
- **Findings:** `exp_a` used `use_volume_weight=True`, which scaled the update vector by `log(number of points passing the gate)`. Since a full scene can have tens of thousands of points, this was multiplying the update magnitude by ~10x *per frame*, completely overwhelming the prototypes.
- **Action:** Created `exp_a_safe` to disable volume weighting and align the update math with the stable `density` baseline, then ran an overnight A/B test.

### Test 3: The Gating Diagnostics (The Breakthrough)
- **Why:** The overnight test revealed that `exp_a_safe` *still* collapsed on Fog and Cross-Sensor corruptions, while `density` excelled. We wrote `check_adaptation_diagnostics.py` to peek into the exact similarity scores the model was seeing during adaptation.
- **Findings:** The diagnostic script exposed two fatal mathematical flaws in Subcluster Gating:
  1. **Shape-Mimicking Noise (The Fog Collapse):** Fog artifacts geometrically resembled the "bus" subclusters, achieving a surprisingly high similarity score of ~`0.73`. Because `exp_a` used a strict `> 0.80` lower-bound gate, the fog particles easily passed through and poisoned the "bus" class (making up 67.7% of all gradients).
  2. **Global Manifold Shifts (The Cross-Sensor Freeze):** Cross-sensor adaptation physically translates the entire point cloud geometry. Because subclusters are exact fixed points in space, this global shift plummeted all similarities down to `~0.25`. The strict `>0.80` gate permanently slammed shut, preventing any adaptation.

## 4. Key Lessons Learned

1. **Prototypes (Hyperplanes) > Subclusters (Points):** 
   Under severe domain shifts, the feature manifold translates. Exact points (subclusters) fail because absolute distances change drastically. Linear Prototypes (hyperplanes) are much more robust because a manifold shift usually preserves its projection onto the directional boundary.
   
2. **Reject Artificially High Confidence:** 
   Corruptions often manifest as highly-confident artifacts (e.g., dense fog clusters). Gating with a strict lower-bound (`> 0.80`) is dangerous. We must use bounded thresholds (e.g., `[0.45, 0.80]`) to explicitly reject artifacts that are "too confident" to be genuine out-of-distribution points.

## 5. The Current Trajectory: `exp_density_hybrid`
Equipped with these geometric proofs, we have officially abandoned frozen Subcluster gating. We have combined the best of both worlds into a new method: **`exp_density_hybrid`**.

This method uses:
- **Prototype Gating** (Hyperplanes) with bounded `[0.45, 0.80]` thresholds to survive global shifts and reject confident noise.
- **Soft Consensus (TTAug)** to leverage extra compute for better confidence weighting.
- **Safe Update Math** (No volume weighting) to prevent gradient explosions.

**Next Steps:** Validate `exp_density_hybrid` on KITTI-C, and if stable, deploy it directly to the ultimate cross-sensor test: KITTI -> NuScenes.

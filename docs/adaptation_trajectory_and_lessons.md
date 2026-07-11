# Unsupervised Adaptation: Trajectory & Lessons Learned

## 1. Project Goal & Initial State
The objective is to build a robust, unsupervised Test-Time Adaptation (TTA) pipeline for LiDAR semantic segmentation, capable of adapting a pre-trained feature extractor to severe synthetic corruptions (KITTI-C) and massive cross-sensor domain shifts (KITTI -> NuScenes).

Our primary experimental method was **`exp_a` (Subcluster-Gated Soft Consensus)**. This method relied on two core concepts:
1. **Soft Consensus (TTAug):** Using geometric augmentations (jitter, scale) to gauge prediction stability.
2. **Subcluster Gating:** Filtering out confident noise by measuring a point's similarity to exact training distribution modes (subclusters) rather than just the class average (prototypes).

## 2. The Problem: Discrepancies in `exp_a` Performance
Early testing produced heavily conflicting results depending on the experimental setup:
- **`test_2_full` (Simulated Sev 2):** Under this regime, `exp_a` cleanly outperformed the density baseline (e.g., +0.0168 on snow, +0.0079 on fog, +0.0041 on motion), with no catastrophic collapse.
- **`unsup_kitti-c` (Pre-generated Sev 3, Overnight Test):** Here, `exp_a` suffered a massive catastrophic collapse, specifically on Fog (plummeting from a 0.0358 baseline down to 0.0019). Meanwhile, the simpler `density` baseline survived and improved slightly.

## 3. Resolving the Contradictions (The Diagnosis)

A deep dive into the code and diagnostic scripts revealed three major confounding factors that warped the overnight test results:

### A. The Threshold Discrepancy (The Fog Collapse)
In the successful `test_2_full` runs, `exp_a` used the default subcluster similarity thresholds of `[0.35, 0.65]`. However, in the `unsup_kitti-c` overnight run, the thresholds were hardcoded to a vastly stricter `[0.80, 0.95]`. 

Diagnostic tests showed that genuine features under Fog corruption rarely score above `0.80`. By slamming the lower bound to `0.80`, the gate blocked almost all genuine adaptation. The *only* points that managed to pass the `>0.80` gate were extreme, confidently-wrong noise artifacts. 

### B. Volume Weighting is Load-Bearing, but Dangerous with Strict Gates
In `test_1_ablation`, removing volume weighting proved disastrous across the board (-0.03 mIoU). Volume weighting is mathematically load-bearing for scaling updates correctly in sparse point clouds. 

However, in the overnight test, combining volume weighting with the broken `[0.80, 0.95]` gate created a fatal feedback loop: the strict gate filtered out all good points leaving only a few dozen confident noise artifacts, and then volume weighting forcefully inflated the gradients of those artifacts, instantly poisoning the prototypes and causing the 0.0019 collapse.

### C. Global Manifold Shifts (The Cross-Sensor Freeze)
While the Fog collapse was a configuration artifact, the Cross-Sensor problem exposed a genuine geometric vulnerability. Cross-sensor adaptation physically translates the entire point cloud geometry. Because subclusters are exact fixed points in absolute space, a global shift plummets all similarities universally (e.g., down to `~0.25`). This freezes adaptation entirely because the points fall far below even the permissive `0.35` gate.

## 4. Key Lessons Learned

1. **Prototypes (Directions) vs. Subclusters (Fixed Points):** 
   Under severe domain shifts (like cross-sensor translation), exact points (subclusters) fail because absolute distances change drastically. Linear Prototypes (hyperplanes) are much more robust to global translation because a manifold shift usually preserves its relative projection onto the directional boundary.
   
2. **Subcluster Gating is Valid, but Highly Threshold-Sensitive:** 
   The initial conclusion that subcluster gating is "mathematically flawed" was incorrect. It works excellently (as proven in `test_2_full`) *provided* the thresholds are permissive enough (`[0.35, 0.65]`) to capture the shifted distribution.

## 5. The Current Trajectory: `exp_density_hybrid`
Equipped with these refined geometric proofs, we have designed **`exp_density_hybrid`** to combine the safest, most robust elements of both methods:

- **Prototype Gating** (Hyperplanes) with bounded `[0.45, 0.80]` thresholds. This inherits the density baseline's robustness to global cross-sensor manifold shifts.
- **Soft Consensus (TTAug)** to leverage extra compute for better confidence weighting.

**Next Steps:** Validate `exp_density_hybrid` on KITTI-C, and carefully monitor the impact of removing volume weighting. If the hybrid method underperforms compared to a properly thresholded `exp_a`, we must reconsider restoring volume weighting and subcluster gating with the correct `[0.35, 0.65]` boundaries.

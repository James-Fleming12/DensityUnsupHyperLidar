# Unsupervised Prototype Update Methods: Ablation Study & Findings

This document tracks the various unsupervised test-time adaptation methods applied to the Hyperdimensional LiDAR model, detailing their mechanisms, empirical outcomes, and failure modes.

## Round 1: Standard Adaptations (Failed to Beat Baseline)
These 4 methods were derived from standard DNN Test-Time Adaptation literature but failed to transfer effectively to the geometric properties of HDC prototypes.

1. **Orthogonalized Exponential Moving Average (EMA)**
   - *Result:* Stagnation (Near zero gains).
   - *Problem:* The update rate was double-scaled by the EMA decay factor ($1 - \alpha$) and the similarity-based effective learning rate, causing the prototype to shift by microscopically small amounts.
2. **Entropy-Gated Filtering**
   - *Result:* Degraded performance (mIoU dropped).
   - *Problem:* In a 13-class dataset, Softmax entropy often naturally hovers around 1.0-2.5. Scaling by $1/\text{entropy}$ severely crippled the learning rate, and hard-gating dropped too many valid samples.
3. **Dynamic Prototype Synergy Bank**
   - *Result:* Minimal gains.
   - *Problem:* The history bank acted as a massive low-pass filter, dragging the model to its source domain and delaying adaptation to the target weather geometry.
4. **Stochastic Source Restoration**
   - *Result:* Minimal gains.
   - *Problem:* The pull-back learning rate ($0.01$) overpowered the forward adaptation rate ($\approx 0.0007$), permanently tethering the model to the unadapted state.

## Round 2: HDC-Specific Geometric Methods (Failed to Beat Baseline)
These 3 methods attempted to leverage the unique subspace geometry of Hyperdimensional Computing but introduced new topological issues.

1. **Contrastive Negative Push (CNP)**
   - *Result:* Marginal accuracy gains, but **degraded mIoU (-0.004)**.
   - *Problem:* Pushing the runner-up prototype likely corrupted the representation of valid neighboring subclusters, destroying geometric boundaries for classes that naturally overlap (e.g., Road vs. Sidewalk).
2. **Dynamic Adaptive Thresholding (DAT)**
   - *Result:* Barely moved (Near zero gains).
   - *Problem:* The threshold of $\mu + \sigma$ was overly strict. It either rejected too many samples, or the batch variance was so high that only noise/outliers exceeded the threshold.
3. **Confidence-Decayed Subcluster Distillation (CDSD)**
   - *Result:* **Huge Accuracy Gain (+0.037 in rain)** but **Huge mIoU Drop (-0.014)**.
   - *Problem:* Distillation forces the master prototype to the geometric center of all its subclusters. Because "Road" and "Background" dominate the subclusters, centering the master prototype improved overall accuracy (due to massive background/road pixel counts). However, for minority classes, empty or noisy subclusters dragged the master prototype out of bounds, destroying their intersection-over-union.

## Round 3: Class-Balancing & Robust Filtering (Failed to Beat Baseline)
These 4 methods attempted to solve the thresholding and distillation issues but proved the Standard Pull's superiority.

1. **Class-Balanced Thresholding (CBT)**
   - *Result:* Dropped Accuracy (-0.017 Night) and mIoU (-0.004 Night).
   - *Problem:* Dynamic per-class EMA thresholds likely became too relaxed for difficult classes, letting highly confident noise drag the prototypes.
2. **Top-K Subcluster Distillation (TKSD)**
   - *Result:* Huge mIoU drop (-0.017 Night, -0.023 Rain).
   - *Problem:* Proves that any form of direct geometric distillation from subclusters destroys the intricate high-dimensional decision boundaries of minority classes. The original Standard Pull correctly treats the master prototype as an independent moving mass rather than a dependent center of mass.
3. **Self-Paced Proportion Pull (SPPP)**
   - *Result:* Dropped Accuracy (-0.009 Night).
   - *Problem:* Forcing a constant 5% update rate meant that in extremely noisy chunks, we forcefully updated using 5% of pure noise, poisoning the prototypes.
4. **Subcluster-Gated Pull (SGP)**
   - *Result:* Slight mIoU drop.
   - *Problem:* Dual-agreement filtering was too strict, preventing the master prototype from escaping its source domain geometry in heavy rain.

## Round 4: Back to Basics
Given the unwavering dominance of the original `inference_update` (Standard Pull), we theorize that its success lies in its K-means-like simplicity: absolute global thresholding, direct master prototype pulling, and momentum buffering. The Round 4 methods are minor variations that maintain this "basic" architecture without over-engineering:

1. **Distance-Weighted Pull (DWP):** Exponentially weights the pull vector by similarity, making highly confident samples pull exponentially harder than borderline samples.
2. **Momentum-Free Direct Pull (MFDP):** Removes the momentum buffer to prevent drag, pulling the prototype purely via gradient descent to the current chunk's mean.
3. **High-Confidence Hard Pull (HCHP):** Sets the base threshold aggressively high (0.70) but multiplies the learning rate by 10x, mimicking sparse but decisive pseudo-labeling.
4. **Multi-Scale Pull (MSP):** Computes a strong pull vector for samples > 0.75 and a weak pull vector for samples > 0.45, combining them for a balanced update.

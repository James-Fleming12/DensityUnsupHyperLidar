# Unsupervised Prototype Update Methods: Ablation Study & Findings

This document tracks the various unsupervised test-time adaptation methods applied to the Hyperdimensional LiDAR model, detailing their mechanisms, empirical outcomes, and failure modes.

## Round 1: Standard Adaptations (Failed to Beat Baseline)
These 4 methods were derived from standard DNN Test-Time Adaptation literature but failed to transfer effectively to the geometric properties of HDC prototypes.

1. **Orthogonalized Exponential Moving Average (EMA)**
   - *Concept:* Standard EMA applied to prototypes, projecting away from negative classes to prevent boundary collapse.
   - *Result:* Stagnation (Near zero gains).
   - *Problem:* The update rate was double-scaled by the EMA decay factor ($1 - \alpha$) and the similarity-based effective learning rate, causing the prototype to shift by microscopically small amounts.
2. **Entropy-Gated Filtering**
   - *Concept:* Scale the learning rate inversely by the Softmax entropy of the prediction.
   - *Result:* Degraded performance (mIoU dropped).
   - *Problem:* In a 13-class dataset, Softmax entropy often naturally hovers around 1.0-2.5. Scaling by $1/\text{entropy}$ severely crippled the learning rate, and hard-gating dropped too many valid samples.
3. **Dynamic Prototype Synergy Bank**
   - *Concept:* Maintain a temporal bank of previous prototypes and update the current prototype as a synergy (weighted average) of the bank.
   - *Result:* Minimal gains.
   - *Problem:* The history bank acted as a massive low-pass filter, dragging the model to its source domain and delaying adaptation to the target weather geometry.
4. **Stochastic Source Restoration**
   - *Concept:* 5% of the time, pull the prototype back to the pristine source domain to prevent catastrophic forgetting.
   - *Result:* Minimal gains.
   - *Problem:* The pull-back learning rate ($0.01$) overpowered the forward adaptation rate ($\approx 0.0007$), permanently tethering the model to the unadapted state.

## Round 2: HDC-Specific Geometric Methods (Failed to Beat Baseline)
These 3 methods attempted to leverage the unique subspace geometry of Hyperdimensional Computing but introduced new topological issues.

1. **Contrastive Negative Push (CNP)**
   - *Concept:* When confident, pull the target class but explicitly *push* the runner-up class away to expand the margin.
   - *Result:* Marginal accuracy gains, but **degraded mIoU (-0.004)**.
   - *Problem:* Pushing the runner-up prototype likely corrupted the representation of valid neighboring subclusters, destroying geometric boundaries for classes that naturally overlap (e.g., Road vs. Sidewalk).
2. **Dynamic Adaptive Thresholding (DAT)**
   - *Concept:* Calculate batch-level $\mu$ and $\sigma$ for similarities, setting a dynamic threshold ($\mu + \sigma$) to adapt to weather-induced global similarity drops.
   - *Result:* Barely moved (Near zero gains).
   - *Problem:* The threshold of $\mu + \sigma$ was overly strict. It either rejected too many samples, or the batch variance was so high that only noise/outliers exceeded the threshold.
3. **Confidence-Decayed Subcluster Distillation (CDSD)**
   - *Concept:* Update subclusters independently, then re-distill the master prototype as the mean of its subclusters.
   - *Result:* **Huge Accuracy Gain (+0.037 in rain)** but **Huge mIoU Drop (-0.014)**.
   - *Problem:* Averaging *all* subclusters properly positioned the master prototype for dominant classes (background/roads) boosting accuracy, but completely destroyed the geometry of minority classes (pedestrians/bikes) whose subclusters might have been empty or noisy.

## Round 3: New Proposed Methods
To address the failures of Round 1 & 2, we propose 4 new methods focusing on class-balancing and robust filtering:

1. **Class-Balanced Thresholding (CBT)**
   - *Concept:* Global thresholds starve minority classes. We maintain a running EMA of similarity *per class*, updating the prototype only if the sample exceeds its specific class's historical average.
2. **Top-K Subcluster Distillation (TKSD)**
   - *Concept:* Fixes the CDSD mIoU drop. We only distill the master prototype from the Top-2 most frequently activated subclusters, ignoring empty or noisy subclusters.
3. **Self-Paced Proportion Pull (SPPP)**
   - *Concept:* Fixes DAT. Instead of statistical thresholds, rank all samples in the chunk and take exactly the Top 5% most confident samples, ensuring a stable, constant adaptation rate regardless of environmental severity.
4. **Subcluster-Gated Pull (SGP)**
   - *Concept:* Dual-agreement filtering. Update the master prototype *only* if the sample has high similarity to the master prototype AND high similarity to its nearest subcluster.

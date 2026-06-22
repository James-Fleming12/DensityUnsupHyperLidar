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

## Round 4: Back to Basics (Failed to Beat Baseline)
Given the unwavering dominance of the original `inference_update` (Standard Pull), we theorize that its success lies in its K-means-like simplicity: absolute global thresholding, direct master prototype pulling, and momentum buffering. The Round 4 methods are minor variations that maintain this "basic" architecture without over-engineering. However, they all failed to beat the baseline (which scored +0.046 Acc in Night and +0.005 Acc in Rain) in severe domain shifts.

1. **Distance-Weighted Pull (DWP):** 
   - *Mechanism:* Exponentially weights the pull vector by similarity, making highly confident samples pull exponentially harder than borderline samples.
   - *Result:* Dropped Night Acc (-0.015) and mIoU (-0.007). Slight gains in Rain (+0.006 Acc, +0.002 mIoU).
   - *Problem:* Exponential weighting gave too much power to ultra-confident noise in severe conditions.
2. **Momentum-Free Direct Pull (MFDP):** 
   - *Mechanism:* Removes the momentum buffer to prevent drag, pulling the prototype purely via gradient descent to the current chunk's mean.
   - *Result:* Big drop in Night Acc (-0.029) and mIoU (-0.015). Minimal effect in Rain (+0.006 Acc, +0.000 mIoU).
   - *Problem:* Removing momentum caused the prototype to violently oscillate into empty space.
3. **High-Confidence Hard Pull (HCHP):** 
   - *Mechanism:* Sets the base threshold aggressively high (0.70) but multiplies the learning rate by 10x, mimicking sparse but decisive pseudo-labeling.
   - *Result:* Massive drop in Night Acc (-0.044) and mIoU (-0.020). Small mIoU bump in Rain (+0.007).
   - *Problem:* Using a high threshold (0.75) completely starved minority classes of updates, leaving them stuck in the source domain.
4. **Multi-Scale Pull (MSP):** 
   - *Mechanism:* Computes a strong pull vector for samples > 0.75 and a weak pull vector for samples > 0.45, combining them for a balanced update.
   - *Result:* Negligible effect. Night (-0.001 Acc, +0.002 mIoU), Rain (+0.003 Acc, +0.007 mIoU).
   - *Problem:* The dual-scale learning rates effectively cancelled each other out, providing no significant advantage over standard pull.

## Round 5: Hybrid Pull & Distillation (Failed to Beat Baseline)
Given that Standard Pull is incredibly stable, and Subcluster Distillation previously provided massive Accuracy gains (but ruined mIoU), we hypothesized that distillation failed because it treated all subclusters equally. If a minority class has 8 empty subclusters and 2 active ones, a raw average drags the master prototype into dead space (destroying mIoU). We proposed 4 new methods that combined the stable Standard Pull with intelligent, weighted Subcluster Distillation. However, they uniformly failed to outpace the baseline in the most severe domain shifts (Night).

1. **Subcluster-Regularized Pull (SRP):** 
   - *Mechanism:* Standard Pull on the master prototype, but the pull vector is an 80/20 mix of the raw sample mean and the geometric mean of the *source subclusters* those samples matched to. This prevents the prototype from drifting off the source manifold.
   - *Result:* Dropped Night Acc (-0.014) and mIoU (-0.005). Moderate gains in Rain (+0.008 Acc, +0.000 mIoU).
   - *Problem:* Regularizing towards the subclusters likely acted as an anchor, preventing necessary geometric adaptation in extreme conditions like Night.
2. **Activity-Weighted Distillation (AWD):** 
   - *Mechanism:* Updates subclusters independently. Instead of a raw mean, it distills the master prototype using an activity-weighted average of the subclusters (weighted by how many samples hit each subcluster in the current chunk).
   - *Result:* Dropped Night Acc (-0.011) and mIoU (-0.010). Moderate gains in Rain (+0.006 Acc, -0.000 mIoU).
   - *Problem:* In heavy noise, the most active subclusters are often the ones absorbing erroneous background points, causing the weighted mean to pull directly into noise.
3. **Confidence-Weighted Distillation (CWD):** 
   - *Mechanism:* Updates subclusters independently. It distills the master prototype by weighting all subclusters based on their cosine similarity to the *current* master prototype (using Softmax). This ignores subclusters that have drifted into noise.
   - *Result:* Stagnation in Night (+0.001 Acc, -0.001 mIoU). Good gains in Rain (+0.008 Acc, +0.001 mIoU).
   - *Problem:* Highly conservative. Weighting by similarity to the master prototype effectively forces the master prototype to only listen to things that already look exactly like it, preventing robust adaptation.
4. **Prototype-Subcluster Ping-Pong (PSP):** 
   - *Mechanism:* Runs Standard Pull on the master prototype. Then, it gently pulls all subclusters for that class towards the *new* master prototype. This keeps the subclusters acting as a flexible, cohesive mesh around the master.
   - *Result:* Dropped Night Acc (-0.010) and mIoU (-0.008). Good gains in Rain (+0.008 Acc, +0.001 mIoU).
   - *Problem:* Pulling subclusters toward the master prototype collapses their natural geometric variance. This "shrinking mesh" loses the ability to classify complex, elongated object shapes.

## Round 6: Advanced Structural Methods (Tradeoff-Based)
In this round, we move beyond simple pulling and distillation to implement more advanced structural constraints and active learning strategies. Each method introduces a specific tradeoff (e.g., compute, memory, or human annotation) in exchange for theoretically stronger performance in edge cases.

1. **Temporal Consistency Gating (TCG):**
   - *Mechanism:* Buffers $N=3$ consecutive LiDAR frames. A target sample is only allowed to exert a Standard Pull on the prototype if the exact same physical space is classified as the same object across all 3 frames. This explicitly filters out transient noise (like heavy rain points) while allowing a lower confidence threshold for capturing static minority classes.
   - *Tradeoff:* Memory & Compute Buffer (requires caching spatial frames).
   - *Result:* Failed. Dropped Night Acc (-0.011) and mIoU (-0.008). Dropped Rain Acc (-0.000) and mIoU (-0.005). 
   - *Analysis:* Delaying updates until 3 consecutive frames agree is too conservative. In highly dynamic scenarios, minority class objects or ego-motion shifts prevent consecutive physical overlap, causing the prototype to completely miss valid updates.

2. **Oracle-Guided Active Anchoring (OGAA):**
   - *Mechanism:* Instead of relying on unsupervised confidence, this method isolates the top-5 most "confusing" samples per chunk (where the margin between the top-1 and top-2 predicted classes is the smallest). It queries an Oracle (ground-truth) for their true labels and executes a massive "Hard Pull" exclusively on these samples.
   - *Tradeoff:* Human Annotation Cost (requires a micro-budget of labeled data in the target domain).
   - *Result:* Catastrophic Failure. Decimated Night Acc (-0.864) and mIoU (-0.123). Decimated Rain Acc (-0.136).
   - *Analysis:* Pulling exclusively on the most confusing, boundary-case samples causes severe prototype collapse. By forcefully anchoring the master prototype to the absolute hardest edge cases, it completely forgets the core geometric center of the class.

3. **Subcluster-Routed Translation (SRT):**
   - *Mechanism:* When a confident sample arrives, it matches to its closest Source Subcluster. We calculate a translation vector $V$ from the Subcluster to the Target Sample. Instead of pulling the Master Prototype directly to the sample, we apply $V$ directly to the Master Prototype. This theoretically retains relative geometric structure.
   - *Tradeoff:* Higher Compute (requires translation vector calculation per-sample against subclusters).
   - *Result:* Failed in Night (-0.011 Acc, -0.005 mIoU), marginal gains in Rain (+0.003 Acc, -0.001 mIoU).
   - *Analysis:* The relative translation vector $V$ assumes the geometric relationship between the subcluster and the master prototype is perfectly rigid across domains. In reality, the entire geometric distribution warps non-linearly, making rigid translations harmful.

4. **Decoupled Memory-Replay Pull (DMRP):**
   - *Mechanism:* Combines the Unsupervised Standard Pull with a continuous Source Domain rehearsal. Every time the prototype is pulled toward a Target Sample, it is simultaneously pulled toward a randomly sampled Source Hypervector from a frozen Replay Buffer (approximated by the subclusters). This continuous "micro-anchoring" guarantees the prototype never drifts entirely into empty space.
   - *Tradeoff:* Replay Buffer Memory Overhead (requires maintaining historical source subclusters).
   - *Result:* Minor success in Rain (+0.011 Acc, +0.002 mIoU), but failed to beat Baseline in Night (+0.006 Acc, -0.002 mIoU).
   - *Analysis:* While the replay buffer prevented catastrophic drift, pulling back to the source domain hindered the massive geometric leaps required to adapt to the severe signal destruction of the Night environment.

### Conclusion on Prototype Updates
Across 6 rigorous rounds of testing complex constraints (Distillations, Temporal Gating, Ping-Pong meshes, Active Anchoring, and Memory Replays), **the original Standard Pull baseline remains the undisputed champion.** Its success lies in its completely unconstrained, aggressive K-means-style plasticity. It does not try to preserve outdated source geometry, nor does it wait for complex temporal confirmations. It simply identifies high-confidence targets and pulls the master prototype directly towards them, allowing the HDC space to freely and massively warp to match the new domain.

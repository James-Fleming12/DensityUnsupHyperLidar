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

### Conclusion on Prototype Updates (Rounds 1-6)
Across 6 rigorous rounds of testing complex constraints (Distillations, Temporal Gating, Ping-Pong meshes, Active Anchoring, and Memory Replays), the original Standard Pull baseline remained the champion due to its unconstrained, aggressive plasticity.

## Round 7: Confidence Calibration & Tradeoffs
This round tested four methods that maintain the gentle momentum of the Standard Pull but introduce specific structural tradeoffs to overcome minority-class starvation and weather noise.

1. **Oracle-Verified Soft Pull (OVSP):**
   - *Mechanism:* We fix the catastrophic failure of Round 6's OGAA. We actively hunt for the 5 "most confusing" samples per chunk and send them to the human Oracle. However, we do not do a Massive Hard Pull. Instead, we assign these 5 samples a synthetic confidence score of 1.0 and feed them into the exact same gentle, momentum-buffered Standard Pull as the rest of the unsupervised pipeline.
   - *Tradeoff:* Target-Domain Labels (Micro-budget).
   - *Result:* Minor gains in Rain (+0.005 Acc, -0.001 mIoU) and Night (+0.004 Acc, +0.003 mIoU).
   - *Analysis:* While it avoided the catastrophic collapse of OGAA, the gentle momentum pull limited the impact of the oracle labels. It underperformed the unconstrained Standard Pull which had the freedom to warp aggressively based purely on confidence.

2. **Density-Calibrated Standard Pull (DCSP):**
   - *Mechanism:* In LiDAR, close objects generate 1,000s of points, while distant objects generate 10s. The baseline Standard Pull treats all these points equally. DCSP scales the pull weight inversely by the point cloud density (radial distance) of the target sample.
   - *Tradeoff:* Compute (Range/Density tracking).
   - *Result:* Massive Accuracy gains! Rain Acc increased by +0.144 and Night Acc by +0.132. Rain mIoU also increased by +0.043. However, Night mIoU dropped by -0.055.
   - *Analysis:* Scaling by range is incredibly effective for raw accuracy. By preventing dense, easy objects from monopolizing the adaptation vector, the prototype manifold is successfully stretched across both close and distant target-domain geometries.

3. **Cross-Augmentation Consistency Gating (CACG):**
   - *Mechanism:* Relies on spatial consistency. For every incoming frame, we create a second, slightly augmented version (e.g., a spatial jitter/roll). If the extracted hypervector for the original frame and the augmented frame predict different classes, it is weather-induced noise and is discarded.
   - *Tradeoff:* Compute (2x Inference per frame).
   - *Result:* Minor gains in Rain (+0.007 Acc, +0.001 mIoU) but slight regressions in Night (-0.006 Acc, -0.005 mIoU).
   - *Analysis:* The gating mechanism was too strict in extremely noisy environments (Night), discarding too many valid samples that suffered from structural jittering, leading to minor regressions compared to the baseline.

4. **Dual-Buffer Memory Replay (DBMR):**
   - *Mechanism:* Maintains two small buffers: a frozen Source Buffer (Sunny) and a dynamic Target Buffer (highly confident Rain/Night samples from recent chunks). The master prototype is simultaneously pulled by the target sample, a Source Buffer sample, and a Target Buffer sample.
   - *Tradeoff:* Memory (2x Buffer Size).
   - *Result:* Flat in Rain (+0.007 Acc, 0.000 mIoU) and regression in Night (-0.001 Acc, -0.008 mIoU).
   - *Analysis:* The continuous "elastic tether" to both the source and target histories stagnated the model, preventing it from making the clean, aggressive breaks needed to adapt to severe weather shifts.

### Conclusion on Prototype Updates (Rounds 1-7)
After 7 exhaustive rounds, **Density-Calibrated Standard Pull (DCSP)** emerged as a highly potent strategy when raw **Accuracy** is the primary metric, netting massive +13-14% accuracy gains over the unadapted baseline by ensuring distant/sparse points pull just as heavily as close/dense points. However, if maintaining strict boundary masks (mIoU) in Night conditions is the priority, the original unconstrained **Standard Pull** remains the most balanced, lowest-compute, and robust method.

## Round 8: Advanced Shock Absorbers & Local Geometric Tethers
This round focused on fixing the tradeoffs from DCSP and testing local/consensus-based adaptation rules.

1. **Class-Normalized Density Clamping (DCSP Fix):**
   - *Mechanism:* Instead of scaling by global density, we scale the sample's density by an EMA of its predicted class's density, clamping the multiplier at a maximum of 1.5x to prevent anomalous sparse points from throwing the prototype into background noise.
   - *Tradeoff:* None.
   - *Result:* Failed to match Standard Pull. Rain Acc (+0.009) and mIoU (+0.001) were negligible, while Night Acc (-0.004) and mIoU (-0.007) regressed.
   - *Analysis:* The strict clamping acted as too harsh of a shock absorber. It prevented the prototype from making the massive geometric leaps required to adapt to severe signal destruction in the Night domain, reverting the massive gains seen in the unconstrained DCSP.

2. **Multi-Jitter Consensus Gating (MJCG):**
   - *Mechanism:* Creates three versions of the input tensor (original, +1 spatial shift, -1 spatial shift). Only pulls the prototype if all three versions independently predict the exact same class, creating a nearly flawless noise filter.
   - *Tradeoff:* 3x Encoding Compute.
   - *Result:* Regression. Rain Acc (-0.004), Night Acc (-0.009), and Night mIoU (-0.007) all dropped.
   - *Analysis:* The 3-way consensus was far too strict. While it successfully filtered out transient weather noise, it also filtered out the legitimate but highly-distorted edge cases that the model desperately needed to learn in order to adapt to the new domain.

3. **K-Nearest Sub-Prototype Pull (KNN-SPP):**
   - *Mechanism:* Instead of dragging the single master prototype, we route the pull vector exclusively to the closest matching local sub-prototype. The master prototype is then re-calculated as the geometric center of its sub-prototypes.
   - *Tradeoff:* Similarity Compute against 65 subclusters.
   - *Result:* Massive Accuracy gains! Rain Acc (+0.144) and Night Acc (+0.035). However, catastrophic mIoU collapse: Rain mIoU (-0.014) and Night mIoU (-0.061).
   - *Analysis:* Adapting local sub-prototypes independently completely destroys the cohesive structural boundary of the class manifold. While it helps classify points deep inside the object (boosting Accuracy), the shattered boundaries decimate the strict mask requirements for semantic segmentation (cratering mIoU).

4. **Two-Pass Distribution Alignment (TPDA):**
   - *Mechanism:* Inference is run on the whole chunk to calculate a class distribution. This is compared to a slow-moving EMA source prior. If a class is wildly over-represented (e.g., a burst of noise looking like Pedestrians), its learning rate is penalized proportionally.
   - *Tradeoff:* Chunk Latency.
   - *Result:* Regression across the board. Night Acc (-0.008) and Night mIoU (-0.008) dropped.
   - *Analysis:* Penalizing "over-represented" classes prevents the model from adapting to massive but legitimate real-world shifts (like a bus dominating the frame for 30 seconds).

### Conclusion on Prototype Updates (Rounds 1-8)
After 8 exhaustive rounds of testing, the core paradigm holds true: **any method that attempts to restrict, gate, clamp, or tether the adaptation process inevitably causes regressions.** The unconstrained, aggressive **Standard Pull** remains the king of mIoU because it allows the HDC manifold to violently warp to the new domain. Methods that boost raw Accuracy (like DCSP or KNN-SPP) do so by destroying the cohesive boundary structure, leading to severe mIoU drops.

## Round 9: Update Normalization & Pacing
This round tested if we could protect minority classes and prevent majority class runaway by equalizing the update rates and volumes across classes.

1. **Equal-Volume Update Queues (EVUQ):**
   - *Mechanism:* Completely disconnects updates from the inference stream. Confident samples are buffered into class-specific FIFO queues (size 100). The prototype is only pulled when the queue is perfectly full, ensuring all classes adapt at the exact same volumetric rate.
   - *Tradeoff:* Memory FIFO.
   - *Result:* Failed. Flatlined in Rain (0.000 Acc, 0.000 mIoU) and regressed in Night (-0.007 Acc, 0.000 mIoU).
   - *Analysis:* Forcing classes to wait for equal volume completely breaks the temporal immediacy needed for adaptation. By the time a rare class fills its queue of 100 samples, the weather conditions or domain lighting may have completely shifted again. It is too slow to react.

2. **Dynamic Class-Paced Momentum (DCPM):**
   - *Mechanism:* Adjusts the learning rate/momentum based on the number of confident samples in the chunk. Massive majority classes (like Road) get highly stiff momentum so they can't runaway, while minority classes get loose momentum to exert maximum leverage from their rare points.
   - *Tradeoff:* Frequency Tracking Compute.
   - *Result:* Modest gains in Rain (+0.007 Acc) and Night (+0.006 Acc, +0.002 mIoU), but drastically underperformed the baseline Standard Pull (+0.055 Night Acc).
   - *Analysis:* Artificially stiffening the momentum of majority classes prevents them from accurately mapping out the massive geometric shifts in the target domain. Majority classes *need* to run away to correctly model the new world.

3. **Prior-Calibrated Similarity Gating (PCSG):**
   - *Mechanism:* Adjusts the raw Cosine Similarity score of a sample based on its class's source training frequency (prior). Rare classes get a massive mathematical boost, artificially forcing their weak, noise-corrupted points over the confidence threshold.
   - *Tradeoff:* None.
   - *Result:* Total Regression. Dropped in Rain (-0.005 Acc, -0.003 mIoU) and Night (-0.007 Acc, -0.006 mIoU).
   - *Analysis:* Artificially lowering the barrier to entry for rare classes is catastrophic. The confidence threshold exists for a reason: it filters out weather noise. By forcing weak minority points through the gate, the minority prototypes were instantly corrupted by rain artifacts and sensor noise.

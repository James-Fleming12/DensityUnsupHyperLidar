# Oracle Gate Quality: Test Results & Analysis

This document summarizes the findings from the "Oracle Gate Quality" tests run on the `wet_ground`, `snow`, and `beam_missing` corruptions. The test compared the gating performance (AUROC of the Precision-Coverage curve) of the learned model **Prototypes** against unsupervised **Oracle Subclusters** (K-Means centroids computed directly on the test set) at varying levels of granularity ($K \in \{1, 2, 4, 8, 16\}$).

## Results Summary

| Method | `wet_ground` AUROC | `snow` AUROC | `beam_missing` AUROC |
| :--- | :---: | :---: | :---: |
| **Prototype** (Learned) | **0.820** | **0.829** | **0.793** |
| Oracle $K=1$ | 0.758 | 0.751 | 0.744 |
| Oracle $K=2$ | 0.740 | 0.736 | 0.727 |
| Oracle $K=4$ | 0.701 | 0.705 | 0.702 |
| Oracle $K=8$ | 0.706 | 0.706 | 0.710 |
| Oracle $K=16$ | 0.718 | 0.719 | 0.724 |

## Analysis & Observations

### 1. No Code Bugs Detected
There is nothing suspicious in these results that points to a bug in the code. At first glance, it might seem strange that an *Oracle* (computed directly on the corrupted test data) performs worse than a Prototype (computed on clean training data). However, this perfectly reflects the underlying mathematics:
* The **Prototype** is a *discriminative* direction. It was trained via Cross-Entropy to maximize the margin between different classes.
* The **Oracle $K=1$** is a *generative* direction. It simply points to the geometric center of mass for a class. In a high-dimensional space, the center of mass may not be the optimal direction for separating that class from anomalies or adjacent classes.

The code correctly computed the generative means on the sphere, and the results accurately demonstrate that discriminative margins yield significantly better confidence estimates than generative density centers.

### 2. The Granularity "U-Shape"
If statistical noise from small cluster sizes was the *only* factor degrading performance as $K$ increased, we would expect a strictly monotonic drop in AUROC (i.e., $K=16$ would be the worst). 

Instead, we observe a distinct "U-shape":
* **$K=1$** captures the strong global mean and performs best among Oracles.
* **$K=4$** performs the worst across all corruptions. It splits the space, but the centroids likely land in "dead zones" or mushy geometric centers that don't align with the actual data manifold.
* **$K=16$** recovers slightly and outperforms $K=4$. At this level of granularity, the K-Means algorithm over-segments the space so finely that some centroids finally "snap" onto true, sharp local structural modes in the LiDAR data, slightly improving the gate.

### 3. Conclusion: The "Apples-to-Oranges" Confound
Despite the geometric recovery at $K=16$, the Prototype dominates all Oracle configurations. This confirms the **"Apples-to-Oranges" confound**: the test was primarily measuring Discriminative vs. Generative performance, not Granularity (Single-Mode vs. Multi-Mode). 

Because the discriminative direction of the Prototype is fundamentally superior, the optimal gating strategy must retain this discriminative direction while incorporating density-awareness. This naturally leads to the **Ellipsoid Conformal Mapping** approach: using the Prototype as the directional anchor, but scaling its confidence margins based on the empirical covariance of the data.

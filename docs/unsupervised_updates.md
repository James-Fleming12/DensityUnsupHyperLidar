# Mathematical Formulations for Unsupervised Test-Time Adaptation in High-Dimensional LiDAR

This document formally outlines the core equations governing the unsupervised update loops in the two primary pipelines: **Density-Guided Test-Time Adaptation** (the Baseline) and **Subcluster-Anchored Soft Consensus Adaptation** (Exp A).

---

## 1. Density-Guided Test-Time Adaptation (Baseline)

The baseline method relies on the standard linear classification prototypes and addresses point cloud sparsity by scaling the influence of points based on their spatial density (or radial distance from the sensor).

### 1.1 Encoding and Inference
Let an input point cloud sweep be $X \in \mathbb{R}^{N \times 3}$. We map the points into a high-dimensional feature space using an encoder $f_\theta$:
$$ H = f_\theta(X) \in \mathbb{R}^{N \times d} $$
The features are unit-normalized to lie on the high-dimensional hypersphere:
$$ Z_i = \frac{H_i}{\|H_i\|_2} $$

Let $P \in \mathbb{R}^{C \times d}$ be the unit-normalized class prototypes (the weights of the final linear classification layer). The model predicts the pseudo-label $\hat{y}_i$ for each point by finding the maximum cosine similarity:
$$ \hat{y}_i = \text{argmax}_{c \in \{1 \dots C\}} (Z_i \cdot P_c) $$
The confidence score $s_i$ is simply the similarity to the predicted prototype:
$$ s_i = Z_i \cdot P_{\hat{y}_i} $$

### 1.2 Gating and Density Weighting
To prevent updating the network on noisy out-of-distribution predictions, a strict thresholding gate $\tau$ is applied:
$$ \mathbb{I}_{gate, i} = \begin{cases} 1 & \text{if } s_i > \tau \text{ and } \hat{y}_i \neq 0 \\ 0 & \text{otherwise} \end{cases} $$
*(Note: $\hat{y}_i = 0$ is the unlabeled/noise class, which is strictly ignored to prevent catastrophic collapse).*

Because LiDAR point clouds are highly imbalanced spatially (dense near the sensor, exponentially sparse at long range), the pseudo-label gradients are weighted by their inverse spatial density. Let $r_i$ be the radial distance (depth) of point $i$, and $\tilde{r}$ be the batch median distance. The spatial weighting factor $w_{dist, i}$ is:
$$ w_{dist, i} = \left( \frac{r_i}{\tilde{r}} \right)^\gamma $$
where $\gamma$ is a scaling hyperparameter (e.g., 3.0).

### 1.3 Prototype Update
The generalized pull vector $\vec{v}_c$ for a class $c$ is the weighted mean of all gated points assigned to $c$:
$$ \vec{v}_c = \frac{1}{\sum_{i: \hat{y}_i=c} \mathbb{I}_{gate, i} w_{dist, i}} \sum_{i: \hat{y}_i=c} \mathbb{I}_{gate, i} w_{dist, i} Z_i $$

The class prototype is then updated via exponential moving average (EMA) with learning rate $\eta$:
$$ P_c^{(t+1)} = \frac{P_c^{(t)} + \eta \vec{v}_c}{\| P_c^{(t)} + \eta \vec{v}_c \|_2} $$

---

## 2. Subcluster-Anchored Soft Consensus Adaptation (Exp A)

The generalized class prototypes in the baseline are highly susceptible to "mode collapse" when facing severe domain gaps (e.g., heavy fog). This method introduces fine-grained Subclusters, multi-view consensus, and a drift anchor to guarantee robust adaptation.

### 2.1 Multi-View Consensus Bundling
Instead of relying on a single view, the input $X$ is subjected to $M$ geometric augmentations (e.g., base, yaw-shifted, and scaled). Let $Z_i^{(m)}$ be the unit-normalized feature encoding for view $m$. 

The views are aggregated into a robust bundled representation:
$$ Z_{bundle, i} = \frac{\sum_{m=1}^M Z_i^{(m)}}{\| \sum_{m=1}^M Z_i^{(m)} \|_2} $$

We define an agreement weight $w_{agree, i} \in [0, 1]$ that scales down the gradient contribution if the predictions across the $M$ views disagree.

### 2.2 Subcluster Gating (Mathematical Core)
During pretraining on the source dataset, the feature space of each class $c$ is clustered into $K$ granular modes, or **Subclusters**: $S_{c, k} \in \mathbb{R}^d$. 
Unlike the highly generalized prototype $P_c$, these subclusters precisely map the topological boundaries of the source distribution.

The confidence score is redefined as the maximum similarity to any valid subcluster of the predicted class:
$$ s_{sub, i} = \max_{k \in \{1 \dots K\}} (Z_{bundle, i} \cdot S_{\hat{y}_i, k}) $$

Instead of a binary hard gate, we use a **Soft Ramp** between $[\tau_{low}, \tau_{high}]$ to provide smooth, continuous gradient scaling:
$$ w_{conf, i} = \max\left(0, \min\left(1, \frac{s_{sub, i} - \tau_{low}}{\tau_{high} - \tau_{low}}\right)\right) $$

The final aggregated weight per point is:
$$ W_i = w_{conf, i} \cdot w_{agree, i} \cdot w_{dist, i} $$

### 2.3 Volume-Weighted Pull and Drift Anchor
The direction vector $\vec{d}_c$ for class $c$ is the weighted center of mass:
$$ \vec{d}_c = \frac{\sum_{i: \hat{y}_i=c} W_i Z_{bundle, i}}{\sum_{i: \hat{y}_i=c} W_i} $$

To prevent small clusters of confident points from pulling the prototype too far, the magnitude of the update is scaled logarithmically by the total mass (Volume Weighting):
$$ \vec{v}_c = \vec{d}_c \cdot \log\left(1 + \sum_{i: \hat{y}_i=c} W_i\right) $$

Finally, to prevent **Catastrophic Forgetting** over long sequences, a static Drift Anchor $A_c$ (a frozen copy of the source prototype $P_c^{(0)}$) exerts a constant restoring force $\lambda$:
$$ P_c^{(t+1)} = \frac{P_c^{(t)} + \eta \vec{v}_c + \lambda (A_c - P_c^{(t)})}{\| P_c^{(t)} + \eta \vec{v}_c + \lambda (A_c - P_c^{(t)}) \|_2} $$

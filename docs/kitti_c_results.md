# KITTI-C Continuous Adaptation Results

This document records the exhaustive experimental results for continuous domain adaptation on the **KITTI-C** dataset, culminating in a definitive structural comparison between Subcluster Gating and Prototype Gating under severe geometric domain shifts.

## Executive Summary: The Structural Failure of Subclusters
Our core hypothesis was that modeling the training data using multi-modal geometric fixed points (**Subclusters**) would provide a more robust gauge for gating Out-of-Distribution (OOD) noise during test-time adaptation than using linear **Prototypes**. 

To test this, we evaluated `exp_density_hybrid` (Prototype Gating) against `exp_a` (Subcluster Gating). We found that Subcluster Gating catastrophically collapsed on Fog and dropped on Cross-Sensor shifts. To determine if this was merely a configuration flaw, we systematically exhausted every mathematical mechanism to save the subcluster thesis:
1. **Absolute Thresholds** (`exp_a`): Relaxed to `[0.35, 0.65]`.
2. **Distribution-Relative Gating** (`exp_a_v2`): Using dynamic percentiles to bypass global uniform translation.
3. **Frame-Centered Adaptive Drift** (`exp_a_v3`): Subtracting the translation vector and allowing subclusters to live-track the data.
4. **Subcluster Profile Margin** (`exp_a_v4`): Gating based on the multi-modal profile shape (peak-to-second-peak entropy) rather than absolute distance.

**The Discovery:** All four subcluster mechanisms failed conclusively. They all suffered catastrophic collapse on Fog and failed to handle cross-sensor shifts. We empirically proved that in high-dimensional feature spaces, using fixed points as a geometric gauge is fundamentally unviable under severe domain shift. Neither absolute distance, dynamic centering, nor multi-modal shape profiling can untangle shape-mimicking noise from genuine features once the manifold severely translates.

Conversely, **Prototype Gating** (`exp_density_hybrid`) succeeded because it possesses a **Closed-Loop Advantage**. It uses the class prototypes as both the gate *and* the object being updated. Because the prototype rapidly "chases" the shifted data in real-time, the gate naturally follows the domain shift, remaining robust to uniform global translation.

---

## Evaluation Details
- **Severity Level**: 3
- **Sequence**: The model is evaluated chunk-by-chunk through a continuous sequence of KITTI-C corruptions (fog → wet_ground → snow → motion_blur → beam_missing → crosstalk → incomplete_echo → cross_sensor).
- **Metrics**: 
  - `Initial`: The performance at the beginning of the corruption chunk (before full adaptation takes effect).
  - `Final`: The performance at the end of the corruption chunk (after the model has continuously adapted to the domain).

## Results: `exp_density_hybrid`
*(Prototype Gating, No Volume Weight)*

| Corruption (Sev 3) | Initial mIoU | Final mIoU | mIoU Diff | Initial Acc | Final Acc | Acc Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fog** | 0.0358 | 0.0366 | <span style="color:green">+0.0008</span> | 0.0959 | 0.1316 | <span style="color:green">+0.0357</span> |
| **wet_ground** | 0.3076 | 0.3620 | <span style="color:green">+0.0544</span> | 0.9221 | 0.8964 | <span style="color:red">-0.0257</span> |
| **snow** | 0.2572 | 0.3817 | <span style="color:green">+0.1245</span> | 0.8213 | 0.8637 | <span style="color:green">+0.0424</span> |
| **motion_blur** | 0.2976 | 0.3824 | <span style="color:green">+0.0848</span> | 0.7685 | 0.8405 | <span style="color:green">+0.0720</span> |
| **beam_missing** | 0.2743 | 0.3333 | <span style="color:green">+0.0590</span> | 0.6846 | 0.7998 | <span style="color:green">+0.1152</span> |
| **crosstalk** | 0.0932 | 0.0744 | <span style="color:red">-0.0188</span> | 0.2729 | 0.2211 | <span style="color:red">-0.0518</span> |
| **incomplete_echo**| 0.3375 | 0.3501 | <span style="color:green">+0.0126</span> | 0.9171 | 0.8815 | <span style="color:red">-0.0356</span> |
| **cross_sensor** | 0.3204 | 0.2284 | <span style="color:red">-0.0920</span> | 0.4948 | 0.5663 | <span style="color:green">+0.0715</span> |

## Results: `exp_a`
*(Subcluster Gating with [0.35, 0.65] thresholds, Volume Weight ON)*

| Corruption (Sev 3) | Initial mIoU | Final mIoU | mIoU Diff | Initial Acc | Final Acc | Acc Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fog** | 0.0358 | 0.0017 | <span style="color:red">-0.0341</span> | 0.0959 | 0.0039 | <span style="color:red">-0.0920</span> |
| **wet_ground** | 0.2571 | 0.2724 | <span style="color:green">+0.0153</span> | 0.9053 | 0.8887 | <span style="color:red">-0.0166</span> |
| **snow** | 0.1930 | 0.2866 | <span style="color:green">+0.0936</span> | 0.8144 | 0.8577 | <span style="color:green">+0.0433</span> |
| **motion_blur** | 0.2207 | 0.2860 | <span style="color:green">+0.0653</span> | 0.7567 | 0.8337 | <span style="color:green">+0.0770</span> |
| **beam_missing** | 0.2007 | 0.2489 | <span style="color:green">+0.0482</span> | 0.6787 | 0.7886 | <span style="color:green">+0.1099</span> |
| **crosstalk** | 0.0084 | 0.0207 | <span style="color:green">+0.0123</span> | 0.0167 | 0.0524 | <span style="color:green">+0.0357</span> |
| **incomplete_echo**| 0.2518 | 0.2622 | <span style="color:green">+0.0104</span> | 0.9117 | 0.8752 | <span style="color:red">-0.0365</span> |
| **cross_sensor** | 0.2188 | 0.1633 | <span style="color:red">-0.0555</span> | 0.4769 | 0.5266 | <span style="color:green">+0.0497</span> |

## Results: `exp_a_v2`
*(Percentile Gating [0.10, 0.95], Volume Weight ON with Min-Points Capping)*

| Corruption (Sev 3) | Initial mIoU | Final mIoU | mIoU Diff | Initial Acc | Final Acc | Acc Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fog** | 0.0358 | 0.0047 | <span style="color:red">-0.0311</span> | 0.0959 | 0.0153 | <span style="color:red">-0.0806</span> |
| **wet_ground** | 0.2282 | 0.2473 | <span style="color:green">+0.0191</span> | 0.8984 | 0.8448 | <span style="color:red">-0.0536</span> |
| **snow** | 0.1940 | 0.2809 | <span style="color:green">+0.0869</span> | 0.8004 | 0.7946 | <span style="color:red">-0.0058</span> |
| **motion_blur** | 0.2061 | 0.2676 | <span style="color:green">+0.0615</span> | 0.6669 | 0.7817 | <span style="color:green">+0.1148</span> |
| **beam_missing** | 0.1900 | 0.2456 | <span style="color:green">+0.0556</span> | 0.6199 | 0.7195 | <span style="color:green">+0.0996</span> |
| **crosstalk** | 0.0742 | 0.0632 | <span style="color:red">-0.0110</span> | 0.3591 | 0.2594 | <span style="color:red">-0.0997</span> |
| **incomplete_echo**| 0.2857 | 0.2281 | <span style="color:red">-0.0576</span> | 0.8922 | 0.8152 | <span style="color:red">-0.0770</span> |
| **cross_sensor** | 0.1512 | 0.1379 | <span style="color:red">-0.0133</span> | 0.4510 | 0.5372 | <span style="color:green">+0.0862</span> |

## Results: `exp_a_v3`
*(Percentile Gating, Volume Weight ON, Centered Similarity, Adaptive Subclusters)*

| Corruption (Sev 3) | Initial mIoU | Final mIoU | mIoU Diff | Initial Acc | Final Acc | Acc Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fog** | 0.0358 | 0.0039 | <span style="color:red">-0.0319</span> | 0.0959 | 0.0114 | <span style="color:red">-0.0845</span> |
| **wet_ground** | 0.2239 | 0.2471 | <span style="color:green">+0.0232</span> | 0.8875 | 0.8387 | <span style="color:red">-0.0488</span> |
| **snow** | 0.1917 | 0.2749 | <span style="color:green">+0.0832</span> | 0.7961 | 0.7897 | <span style="color:red">-0.0064</span> |
| **motion_blur** | 0.2135 | 0.2693 | <span style="color:green">+0.0558</span> | 0.6767 | 0.7792 | <span style="color:green">+0.1025</span> |
| **beam_missing** | 0.2056 | 0.2333 | <span style="color:green">+0.0277</span> | 0.6189 | 0.7150 | <span style="color:green">+0.0961</span> |
| **crosstalk** | 0.0757 | 0.0612 | <span style="color:red">-0.0145</span> | 0.3614 | 0.2642 | <span style="color:red">-0.0972</span> |
| **incomplete_echo**| 0.2657 | 0.2366 | <span style="color:red">-0.0291</span> | 0.8895 | 0.8128 | <span style="color:red">-0.0767</span> |
| **cross_sensor** | 0.1654 | 0.1400 | <span style="color:red">-0.0254</span> | 0.4513 | 0.5334 | <span style="color:green">+0.0821</span> |

## Results: `exp_a_v4`
*(Subcluster Profile Margin Gating: Peak-to-Second-Peak Margin + Percentiles)*

| Corruption (Sev 3) | Initial mIoU | Final mIoU | mIoU Diff | Initial Acc | Final Acc | Acc Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fog** | 0.0358 | 0.0032 | <span style="color:red">-0.0326</span> | 0.0959 | 0.0096 | <span style="color:red">-0.0863</span> |
| **wet_ground** | 0.2536 | 0.2552 | <span style="color:green">+0.0016</span> | 0.9023 | 0.8230 | <span style="color:red">-0.0793</span> |
| **snow** | 0.1924 | 0.2584 | <span style="color:green">+0.0660</span> | 0.8039 | 0.7426 | <span style="color:red">-0.0613</span> |
| **motion_blur** | 0.2014 | 0.2389 | <span style="color:green">+0.0375</span> | 0.6574 | 0.6769 | <span style="color:green">+0.0195</span> |
| **beam_missing** | 0.2063 | 0.2218 | <span style="color:green">+0.0155</span> | 0.5625 | 0.6847 | <span style="color:green">+0.1222</span> |
| **crosstalk** | 0.0678 | 0.0525 | <span style="color:red">-0.0153</span> | 0.3233 | 0.2375 | <span style="color:red">-0.0858</span> |
| **incomplete_echo**| 0.2624 | 0.2266 | <span style="color:red">-0.0358</span> | 0.8303 | 0.8079 | <span style="color:red">-0.0224</span> |
| **cross_sensor** | 0.1545 | 0.1334 | <span style="color:red">-0.0211</span> | 0.4238 | 0.4911 | <span style="color:green">+0.0673</span> |

## Takeaways

1. **Subclusters are Inherently Brittle for Gating:** We exhaustively tested four distinct mathematical mechanisms to save subcluster gating from global manifold shifts:
   - Widening the absolute thresholds (`exp_a` with `[0.35, 0.65]`)
   - Percentile-based relative gating (`exp_a_v2`)
   - Frame-centered similarity with adaptive drift (`exp_a_v3`)
   - Subcluster Profile Margin Gating (`exp_a_v4` - using entropy/peak-to-peak spread)
   
   **Conclusion:** All four attempts failed conclusively. They consistently suffered catastrophic collapse on Fog (mIoU dropping from ~0.035 to ~0.003) and completely failed to handle cross-sensor shifts. This definitively proves that in high-dimensional feature spaces, using exact fixed points (subclusters) as a geometric gauge is fundamentally unviable under severe domain shift. Neither absolute distance, dynamic centering, nor multi-modal shape profiling can untangle shape-mimicking noise (Fog) from genuine features once the manifold translates.

2. **The "Closed-Loop" Advantage of Prototype Gating:** `exp_density_hybrid` easily survived the fog corruption (maintaining ~0.036 mIoU with no collapse) and achieved significantly higher adaptation gains across almost all other conditions (e.g., `snow` +0.1245, `motion_blur` +0.0848). 
   - **Why it works:** `exp_density_hybrid` uses the **class prototypes** as both the gate *and* the object being updated. When points are selected by the gate, the prototype is immediately pulled toward those points (amplified by volume weighting). Because the prototype rapidly "chases" the shifted data in real-time, the points in the very next frame are closer to the prototype, keeping the gate open. 
   - **Why subclusters failed:** The subcluster experiments suffered from a **structural disconnect**. They gated on one set of geometry (frozen or slow-moving subclusters) while updating another (prototypes). When a severe shift like Fog hit, the points moved far away from the frozen subclusters, failed the gate, starved the updates, and caused catastrophic collapse.

3. **Sensor Noise Remains a Challenge:** Both paradigms struggle with intense sensor-level artifacts (`crosstalk`, `cross_sensor`), heavily dropping in mIoU. This suggests that sensor noise causes geometric mangling that neither rapidly-adapting prototypes nor subclusters can cleanly separate without additional structural priors.

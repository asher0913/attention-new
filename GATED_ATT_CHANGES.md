# Gated-Att Baseline vs. Original CEM-main

This note documents every code-level change applied when cloning the original `CEM-main` project into `gated-att/` and replacing the Gaussian-mixture surrogate with the gated-attention variant used in the CVPR'18 MIL paper. The goal is to ensure the Linux `run_exp.sh` pipeline executes the pure gated-attention experiment without any residual GMM/KMeans logic.

## Overview

- **Source snapshot**: `gated-att/` started as a verbatim copy of `CEM-main/`.
- **Objective**: drop all Gaussian-mixture / KMeans conditional-entropy machinery and replace it with the gated-attention surrogate (`GatedAttentionCEM`) while preserving the rest of the training / attack infrastructure.
- **Execution guarantees**: `gated-att/run_exp.sh` now changes into its own directory before launching `python main_MIA.py`, so invoking the script from Linux runs the gated-att pipeline by default.

## Detailed Changes (by file)

### `gated-att/model_training_paral_pruning.py`
- **New modules**: introduced `GatedAttentionPooling` and `GatedAttentionCEM` (lines ~34-116) implementing the exact gated-attention formulation from Ilse et al. (2018). These compute per-class attention weights, weighted means, and log-variance surrogate losses.
- **Import cleanup**: removed all GMM/KMeans related imports (`GMM.fit_gmm_torch`, `GaussianMixture`, `KMeans`, `PCA`, etc.), leaving only the dependencies required by gated attention.
- **State initialisation**: dropped legacy `self.use_attention_cem` / centroid caches, keeping only `self.gated_attention_cem = None` for lazy instantiation (line ~432).
- **Training step rewiring**:
  - `train_target_step` signature simplified to `(x_private, label_private, adding_noise, random_ini_centers, client_id)` (line ~965).
  - Inside the step, encoder features are flattened and fed to `GatedAttentionCEM`; NaN/Inf guards reset the loss to zero if the batch carries invalid values (lines ~987-1015).
  - Removed all centroid/KMeans updates, distance matrices, and `compute_class_means`; the gated surrogate provides `rob_loss` directly (lines ~1090-1233).
  - Preserved the dual-backprop pipeline (first backprop through `rob_loss`, store encoder grads, then classify and combine) so the regularisation strength `lambd` still modulates encoder updates.
- **Scheduler / epoch loop**:
  - Deleted the entire post-epoch “feature harvesting + KMeans” phase (former lines ~1680-1890).
  - Removed helper methods `kmeans_plusplus_init`, `kmeans_cuda`, and `apply_gmm_with_pca_and_inverse_transform`, along with all references to centroid buffers.
  - Cleaned up leftover comments and legacy debug output so no GMM code remains reachable.

### Removed legacy files
- Deleted `gated-att/GMM.py`, `gated-att/model_training_GMM.py`, and `gated-att/model_training.py` to prevent stale imports and guarantee the project cannot fall back to the old surrogate.

### `gated-att/main_MIA.py` & `gated-att/main_test_MIA.py`
- Dropped the unused `import model_training` statement; both scripts now import only `model_training_paral_pruning`, ensuring Python never searches for the removed GMM modules.

### `gated-att/run_exp.sh`
- Re-enabled `cd "$(dirname "$0")"` at the top so running `bash run_exp.sh` from Linux always executes within `gated-att/`, guaranteeing the gated-attention codepath is launched.

## How to Run (Linux)

```bash
cd gated-att
bash run_exp.sh
```

The script calls `python main_MIA.py ...` inside `gated-att/`, which in turn instantiates the new `GatedAttentionCEM` surrogate. No GMM artefacts remain in the call chain, so experiments isolate the gated-attention behaviour.

## Summary of Behavioural Differences

| Area | CEM-main (original) | gated-att (this repo) |
| ---- | ------------------- | --------------------- |
| CEM surrogate | Per-class KMeans + GMM log-variance | Weighted gated attention (Ilse et al.) |
| Extra buffers | Centroid caches, covariance logs, slot visualisation | Removed (only attention weights kept) |
| Epoch tail-step | Full-batch feature extraction + clustering each epoch | Deleted; gated attention operates on mini-batches |
| Scripts | Mixed imports (`model_training`, `model_training_paral_pruning`) | Imports trimmed to gated-att module only |
| Filesystem | Contains `GMM.py`, `model_training_GMM.py`, centroid utilities | Files removed to avoid accidental reuse |

This document should help track every delta from `CEM-main` and justify why the changes were required for the gated-attention experiment. If additional attention variants or ablations are needed, they can now be implemented directly on top of `GatedAttentionCEM` without reintroducing GMM dependencies.

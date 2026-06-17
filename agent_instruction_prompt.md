# Prompt for AI Coding Agent: Implement Multi-Stage Training, Sweep, and Evaluation Pipeline

Use the instructions below to implement a complete, robust training and evaluation pipeline for object detection models in a target codebase (e.g., YOLO12 or other libraries).

---

## Objective
Design and implement a Python-based machine learning pipeline that supports:
1. **Multi-Seed Stock Training**: Training a set of models across multiple seeds with W&B logging.
2. **Two-Stage Hyperparameter Optimization (HPO)** via W&B Sweeps:
   - **Phase 1 (Augmentation Tuning)**: Optimizing custom data augmentations (e.g., Albumentations, mosaic, mixup) while locking learning hyperparameters.
   - **Phase 2 (Learning HPO)**: Retrieving the best Phase 1 augmentation settings and optimizing optimizer settings (learning rate, weight decay, momentum, optimizer choice).
3. **Strict COCO Evaluation**: Running predictions on a test split, mapping IDs, and calculating average precision/recall using `pycocotools.cocoeval`.
4. **Metrics Aggregation & Plotting**: Collecting metrics across seeds, calculating mean ± standard deviation, and plotting results.
5. **Memory Management**: Preventing GPU memory accumulation when training/evaluating sequentially in a loop.

---

## Target Architecture

You need to create or modify five core modules:

### 1. Unified Configuration (`config.py`)
Create a single source of truth class (using `@dataclass` or similar) containing:
- W&B entity, project name, and directory.
- Model variant lists and seed lists (e.g., `42, 100, 999`).
- Dataset YAML/config paths.
- Global training constraints: image size, batch size, epochs (both sweep and production versions), device, and dataloader workers.
- A static method/utility to retrieve the configuration of the best run from a finished W&B sweep using the W&B API:
  ```python
  import wandb
  api = wandb.Api()
  sweep = api.sweep("entity/project/sweep_id")
  # sort runs by a target metric (e.g. metrics/mAP50-95(B)) and retrieve the best config
  ```

### 2. Multi-Stage Sweep Runner (`run_sweep.py`)
Support two phases determined by a `phase` parameter in the sweep config:
- **Phase 1 ("augmentation")**: Lock learning rate, weight decay, optimizer, momentum, and warmup epochs. Dynamically construct a custom augmentation pipeline (e.g., using Albumentations) using probabilities injected by the sweep agent.
- **Phase 2 ("hpo")**: Programmatically fetch the best augmentation config from Phase 1 using the W&B API. Pass the optimized augmentation pipeline to the training settings, and sweep over optimizer parameters (`optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`).
- Ensure `wandb.finish()` is cleanly called at the end of each run.

### 3. Stock Training Runner (`run_training.py`)
- Run sequential training loops for the list of model variants and seeds.
- Construct descriptive run names, e.g., `{model_name}_seed_{seed}_best_aug_hpo`.
- Support loading optimized configurations dynamically if augmentation or HPO sweep IDs are provided.
- Initialize `wandb.init()` with `reinit=True`.
- **Memory Cleanup**: Implement a `finally` block in the loop to delete the model reference, invoke Python's garbage collector, and empty PyTorch's CUDA cache:
  ```python
  if "model" in locals():
      del model
  import gc, torch
  gc.collect()
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  ```
- Trigger the strict COCO evaluation script automatically after training completes.

### 4. Evaluation and Mapping Utilities (`eval_utils.py` & `run_evaluation.py`)
Ultralytics-style or custom framework validators often export predictions using internal image string IDs. Implement a strict evaluation module:
- **COCO Ground Truth Generator**: Parse the dataset YAML, scan the target split (e.g., `test`), and build/save a valid COCO format JSON mapping image filenames to integer IDs. Skip if it already exists.
- **Prediction Mapping**: Match prediction bounding boxes to the integer ground truth IDs.
- **Strict COCO Evaluation**: Load the ground truth and mapped predictions using `pycocotools.coco.COCO` and run `COCOeval(coco_gt, coco_dt, iouType="bbox")` to get standardized AP/AR metrics.
- Save the final metrics in a JSON file matching `{run_name}_coco_metrics.json`.
- Apply GPU memory cache clearing after every evaluation in the evaluation loop.

### 5. Aggregation & Plotting (`plot_results.py`)
- Read all evaluation metric files (`*_coco_metrics.json`) in the results folder.
- Group metrics by model type, stripping seed suffixes.
- Calculate **Mean ± Standard Deviation** across the seeds for each model.
- Generate publication-quality comparative bar plots (using `matplotlib`/`seaborn` with standard error bars):
  - **AP comparison plot (`ap_metrics_comparison.png`)**: AP, AP50, and AP75 side-by-side to show overall performance and regression strictness.
  - **Scale-wise AP plot (`ap_scale_comparison.png`)**: AP_small, AP_medium, and AP_large to analyze target scale detection capabilities.
  - **Recall comparison plot (`ar_metrics_comparison.png`)**: AR@1, AR@10, and AR@100 to show the completeness of target retrieval.
  - **Seed spread distribution (`ap_seed_distribution.png`)**: Swarmplot showing individual points to evaluate variance and model stability.
- Write out a markdown summary table summarizing performance across all 12 COCO metrics:
  ```markdown
  | Model Variant | Runs | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | AP (Small) | AP (Medium) | AP (Large) | AR@100 |
  | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | model_a | 3 | 0.3542 ± 0.0012 | 0.5421 ± 0.0023 | ... | ... | ... | ... | ... |
  ```

---

## Coding Standards
- Preserve all existing file paths and workspace dependencies.
- Ensure proper exception handling in training loops so a failure in one model or seed doesn't crash the entire sequence.
- Keep comments explaining hyperparameter routing clearly documented.

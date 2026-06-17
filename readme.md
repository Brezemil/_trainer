# Multi-Stage YOLO HPO & Training Pipeline with W&B

This repository contains a unified environment for training, hyperparameter optimization (HPO), strict COCO evaluation, and plotting for Ultralytics YOLO models (specifically supporting YOLO11, YOLO26, and RT-DETR) using Weights & Biases (W&B) and Albumentations.

---

## 📋 Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [CLI Reference (Available Commands)](#cli-reference-available-commands)
3. [Configuration Single Source of Truth (`config.py`)](#configuration-single-source-of-truth-configpy)
4. [Unified HPO Sweep Pipeline (Stage 1 & 2)](#unified-hpo-sweep-pipeline-stage-1--2)
5. [Stock Settings Training Suite](#stock-settings-training-suite)
6. [Evaluation & Plotting Suite](#evaluation--plotting-suite)
7. [Codebase API Reference (Detailed Functions)](#codebase-api-reference-detailed-functions)

---

## 🔧 Prerequisites & Setup

Ensure you have the required packages installed and are logged into your Weights & Biases account:

```bash
pip install ultralytics wandb albumentations pycocotools
wandb login
```

### File Structure

```text
├── config.py            # Centralized configuration paths and helpers
├── eval_utils.py        # YOLO-to-COCO translation and pycocotools evaluation
├── run_sweep.py         # Unified execution script for both sweeps
├── run_training.py      # Script to run stock training + evaluation
├── run_evaluation.py    # Script to run evaluation on existing checkpoints
├── plot_results.py      # Script to aggregate metrics across seeds and plot
├── sweep_aug.yaml       # Phase 1 configuration (Augmentations)
└── sweep_hpo.yaml       # Phase 2 configuration (Learning Parameters)
```

---

## 💻 CLI Reference (Available Commands)

All tasks are registered in `pixi.toml` and can be invoked using the `pixi run <command>` syntax:

### 🛠️ Hardware & Environment Checks
* **`pixi run check-gpu`**: Verifies PyTorch/CUDA installation and GPU availability on the system.
* **`pixi run check-yolo`**: Runs the Ultralytics environment diagnosis checks.
* **`pixi run wandb-login`**: Prompts the W&B API login flow to link your machine.

### 🚀 Training Command
* **`pixi run train`**: Runs the entire training suite. By default, it trains YOLO11s, YOLO26s, and RT-DETR-l sequentially, 3 times each, using fixed seeds ($42, 100, 999$). Each run is immediately evaluated on the test set using `pycocotools`.
  * *Parameters:*
    * `--model`: Train a specific model only (e.g. `yolo11s.pt`).
    * `--seed`: Train with a specific seed only (e.g. `42`).
    * `--epochs`: Override default epochs (e.g. `20`).
    * `--batch`: Override default batch size (e.g. `32`).
    * `--device`: Override device (e.g. `0` or `cpu`).
    * `--imgsz`: Override image size (e.g. `256`).
    * `--workers`: Override dataloader workers count.
    * `--fraction`: Override dataset fraction (e.g., `0.01` for a 1% data smoketest).
    * `--runs-dir`: Save runs under a custom folder.
    * `--wandb-dir`: Save W&B metadata files under a custom folder.
  * *Example:*
    ```bash
    pixi run train --model yolo26s.pt --seed 42 --epochs 50 --batch 32
    ```

### 📊 Standalone Evaluation
* **`pixi run eval`**: Runs strict COCO evaluation on existing checkpoints (`best.pt`) found under `runs/{run_name}/`.
  * *Parameters:* Accepts the same command-line parameter overrides as the training script (e.g. `--model`, `--seed`, `--runs-dir`).
  * *Example:*
    ```bash
    pixi run eval --model yolo11s.pt --seed 100
    ```

### 📈 Aggregation & Plotting
* **`pixi run plot`**: Reads all saved metric JSON files from the evaluation results directory, aggregates seeds (mean ± std) per model variant, writes a summary markdown table, and generates comparison charts.

---

## 🛠️ Configuration Single Source of Truth (`config.py`)

All global parameters, models to run, seeds, and logging directories are centralized in `config.py` using `PipelineConfig`:

* **`entity`** / **`project`**: Your W&B usernames/projects.
* **`models`**: Tuple of model weights to train/evaluate (default: `("yolo11s.pt", "yolo26s.pt", "rtdetr-l.pt")`).
* **`seeds`**: Tuple of fixed seeds to run (default: `(42, 100, 999)`).
* **`fraction`**: Default dataset fraction (default: `1.0`).
* **`runs_dir`**: Folder where model checkpoints are saved.
* **`wandb_dir`**: Folder where W&B logs are saved.
* **`eval_results_dir`**: Folder where validation results are saved.
* **Path Resolution Rules:** Relative paths inside `config.py` automatically resolve relative to the project root directory containing `config.py`. Absolute paths are used as-is, making configuration across different PCs easy.

---

## 🔄 Unified HPO Sweep Pipeline (Stage 1 & 2)

Sweeps use W&B's server-side Bayesian search to optimize augmentation (Stage 1) and learning parameters (Stage 2).

### Stage 1: Augmentation Sweep
Locks learning parameters to baselines while searching over geometric/color augmentations.
1. Initialize the sweep with W&B:
   ```bash
   wandb sweep sweep_aug.yaml
   ```
2. Note the generated **Stage 1 Sweep ID** (e.g. `abc123xyz`).
3. Start local sweep agents to execute training:
   ```bash
   wandb agent <entity>/<project>/abc123xyz
   ```

### Stage 2: HPO Sweep
Retrieves the best augmentation configuration from Stage 1, locks them, and sweeps across learning parameters (e.g. learning rate, momentum, optimizer).
1. Open `sweep_hpo.yaml` and update the `prev_aug_sweep_id` parameter value to your Stage 1 Sweep ID:
   ```yaml
   prev_aug_sweep_id:
     value: "abc123xyz"
   ```
2. Initialize and run:
   ```bash
   wandb sweep sweep_hpo.yaml
   wandb agent <entity>/<project>/<hpo_sweep_id>
   ```

---

## 📚 Codebase API Reference (Detailed Functions)

### 📄 `config.py`

#### `class PipelineConfig` (Dataclass)
Defines all global parameters, dataset paths, training parameters, model types, seeds, and logging targets.
* **`__post_init__(self)`**:
  - Initializes baseline fixed loss dictionary `{"box": 5.63, "cls": 0.56, "dfl": 9.04}`.
  - Automatically resolves `runs_dir`, `wandb_dir`, and `eval_results_dir` to absolute paths relative to the folder containing `config.py` if they are defined as relative paths.
* **`get_best_sweep_config(sweep_id: str, project: str, entity: str) -> dict`** (Static Method):
  - Fetches the hyperparameter configuration of the best-performing run (sorted by `metrics/mAP50-95(B)`) from a completed W&B sweep using the W&B API. Strips internal keys starting with `_`.

---

### 📄 `eval_utils.py`

#### `parse_dataset_yaml(dataset_yaml_path: str) -> dict`
Parses the dataset configuration YAML file and returns its content as a Python dictionary.

#### `generate_coco_gt(dataset_yaml_path: str, split: str, save_path: str) -> str`
Converts a split (e.g. `test` or `val`) of a YOLO-formatted dataset (containing images and `.txt` label files) into a single, standard COCO ground-truth JSON file.
- Checks if the file already exists at `save_path` and skips generation if so.
- Maps YOLO 0-indexed categories to COCO 1-indexed categories (`cls_id + 1`).
- Resolves relative paths defined in `dataset.yaml` to absolute paths.
- Computes absolute pixel bounding boxes `[x_min, y_min, width, height]` from normalized center coordinates.

#### `evaluate_model_coco(...)`
Main evaluation wrapper function. Runs the following steps:
1. Resolves `eval_results_dir` to an absolute path.
2. Instantiates the model (uses `RTDETR` if the weights name contains `rtdetr`, otherwise `YOLO`).
3. Runs the Ultralytics validation logic: `model.val()` with `save_json=True` in a temporary folder.
4. Generates the split ground-truth COCO JSON using `generate_coco_gt()`.
5. Loads `predictions.json`, maps string image IDs to integer IDs, and saves them to a mapped JSON file.
6. Initializes `pycocotools.coco.COCO` and runs `pycocotools.cocoeval.COCOeval` to compute strict AP/AR metrics.
7. Saves results as `{run_name}_coco_metrics.json`.
8. Deletes the temporary validation directory to keep the workspace clean.

---

### 📄 `run_sweep.py`

#### `build_albumentations_pipeline(config_dict: dict, imgsz: int) -> list`
Constructs an active Albumentations transformation pipeline (e.g. Rotate, Flips, GridDistortion, Sharpen, ISONoise) using execution probabilities passed from the W&B sweep agent. Returns a list of transformation objects.

#### `main()`
Entry point for sweep training runs. Configures settings, initializes W&B, disables native YOLO augmentations (to evaluate Albumentations custom probabilities in isolation), resolves the sweep phase (`augmentation` or `hpo`), instantiates the model, executes training, and finishes cleanly.

---

### 📄 `run_training.py`

#### `parse_args() -> argparse.Namespace`
Sets up the command-line argument parser, defining options to override default models, seeds, epochs, batches, image sizes, devices, dataloader workers count, dataset fraction, runs directory, and wandb directory.

#### `main()`
Primary training execution logic:
1. Iterates over selected models and seeds.
2. Formats a unique `run_name` (`{model_base}_seed_{seed}`).
3. Initializes a W&B run targeting the configurable `wandb_dir`.
4. Instantiates `YOLO` or `RTDETR` and trains the model saving checkpoints to `runs_dir`.
5. Immediately calls `evaluate_model_coco()` from `eval_utils` to run strict COCO test evaluation.
6. Shuts down the W&B run cleanly.

---

### 📄 `run_evaluation.py`

#### `parse_args() -> argparse.Namespace`
Sets up CLI arguments for standalone evaluation (accepts the same overrides as the training script).

#### `main()`
Checks the configured `runs_dir` for training checkpoints (`runs/{run_name}/weights/best.pt`) and executes the strict COCO validation split evaluation on them sequentially without retraining.

---

### 📄 `plot_results.py`

#### `load_metrics(eval_dir: str) -> list`
Loads all JSON metric files matching `*_coco_metrics.json` inside the evaluation results directory.

#### `group_and_aggregate(results: list) -> dict`
Groups metric dictionary objects by model variant base name, parses their seeds, and calculates the mean and standard deviation for each COCO metric.

#### `plot_ap_comparison(stats: dict, save_dir: str)`
Generates a bar plot comparing mAP@0.50:0.95 and mAP@0.50 across model variants. Includes error bars reflecting the standard deviation across seeds. Saves to `ap_metrics_comparison.png`.

#### `plot_individual_seeds(stats: dict, save_dir: str)`
Generates a strip/swarm plot showing individual point scores for each seed run to visualize variance. Saves to `ap_seed_distribution.png`.

#### `generate_markdown_summary(stats: dict, save_dir: str)`
Renders a markdown table detailing the aggregated mean ± std scores for all 12 COCO metrics across the models, saving it to `evaluation_summary.md`.

#### `main()`
Entry point for plotting. Loads metrics, groups them, runs the plotting routines, and writes the summary table.
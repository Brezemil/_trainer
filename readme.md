# Multi-Stage YOLO HPO & Training Pipeline with W&B

This repository contains a unified environment for training, hyperparameter optimization (HPO), strict COCO evaluation, and plotting for Ultralytics YOLO models (specifically supporting YOLO11, YOLO26, and RT-DETR) using Weights & Biases (W&B) and Albumentations.

---

## 📋 Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [CLI Reference (Available Commands)](#cli-reference-available-commands)
3. [Configuration Single Source of Truth (`config.py`)](#configuration-single-source-of-truth-configpy)
4. [Unified HPO Sweep Pipeline (Stage 1 & 2)](#unified-hpo-sweep-pipeline-stage-1--2)
5. [Training, Evaluation, and Plotting Workflow](#training-evaluation-and-plotting-workflow)
6. [Codebase API Reference (Detailed Functions)](#codebase-api-reference-detailed-functions)

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
    * `--aug-sweep-id`: W&B sweep ID from Phase 1 (Augmentation Tuning) to plug in.
    * `--hpo-sweep-id`: W&B sweep ID from Phase 2 (HPO) to plug in.
  * *Plug-in Sweep Naming Convention (W&B):*
    * If only `--aug-sweep-id` is provided, runs are named: `{model_base}_seed_{seed}_best_aug`
    * If only `--hpo-sweep-id` is provided, runs are named: `{model_base}_seed_{seed}_best_hpo`
    * If both are provided, runs are named: `{model_base}_seed_{seed}_best_aug_hpo`
  * *CLI Examples (Common Use-Cases):*
    * **Train YOLO26s three times (across all three seeds):**
      ```bash
      pixi run train --model yolo26s.pt
      ```
    * **Train YOLO26m three times (across all three seeds):**
      ```bash
      pixi run train --model yolo26m.pt
      ```
    * **Train YOLO26s only one time (with a specific seed):**
      ```bash
      pixi run train --model yolo26s.pt --seed 42
      ```
    * **Train all default models (YOLO11, YOLO26, RT-DETR) across all seeds:**
      ```bash
      pixi run train
      ```
    * **Override training parameters (e.g., 50 epochs, batch 32, image size 256):**
      ```bash
      pixi run train --model yolo26s.pt --seed 100 --epochs 50 --batch 32 --imgsz 256 --device 0
      ```

### 📊 Standalone Evaluation
* **`pixi run eval`**: Runs strict COCO evaluation on existing checkpoints (`best.pt`) found under `runs/{run_name}/`.
  * *Parameters:* Accepts the same command-line parameter overrides as the training script (e.g. `--model`, `--seed`, `--runs-dir`, `--aug-sweep-id`, `--hpo-sweep-id`).
  * *Example (Evaluating runs trained with both sweeps plugged in):*
    ```bash
    pixi run eval --model yolo26s.pt --seed 42 --aug-sweep-id abc123xyz --hpo-sweep-id hpo456uvw
    ```

### 📈 Aggregation & Plotting
* **`pixi run plot`**: Reads all saved metric JSON files from the evaluation results directory, aggregates seeds (mean ± std) per model variant, writes a summary markdown table, and generates comparison charts.

---

## 🛠️ Configuration Single Source of Truth (`config.py`)

All global parameters, models to run, seeds, and logging directories are centralized in `config.py` using `PipelineConfig`:

* **`entity`** / **`project`**: Your W&B usernames/projects.
* **`models`**: Tuple of model weights to train/evaluate (default: `("yolo11s.pt", "yolo26s.pt", "rtdetr-l.pt")`).
* **`seeds`**: Tuple of fixed seeds to run (default: `(42, 100, 999)`).
* **`fraction`**: Default dataset fraction (default: `1.0`, set to `0.01` to run a 1% dataset smoketest).
* **`runs_dir`**: Folder where model checkpoints are saved.
* **`wandb_dir`**: Folder where W&B logs are saved.
* **`eval_results_dir`**: Folder where validation results are saved.
* **`aug_sweep_id`**: Optional default W&B sweep ID for Phase 1 (Augmentation).
* **`hpo_sweep_id`**: Optional default W&B sweep ID for Phase 2 (HPO).
* **Path Resolution Rules:** Relative paths inside `config.py` automatically resolve relative to the project root directory containing `config.py`. Absolute paths are used as-is, making configuration across different PCs easy.

---

## 🔄 Unified HPO Sweep Pipeline (Stage 1 & 2)

Sweeps use W&B's server-side Bayesian search to optimize augmentation (Stage 1) and learning parameters (Stage 2).

### 🎨 Hybrid Augmentation Strategy

For complex domains like UAV tree canopy detection, standard YOLO augmentations are augmented with a robust **Albumentations** pipeline specifically optimized for nadir (top-down) aerial imagery:
* **Native YOLO Augmentations:** Handles multi-image stitching and label alignment (`mosaic` is kept high for small canopy detection, `mixup` is disabled to prevent blending of tree edges, and `copy_paste` is restricted to avoid unrealistic overlaps).
* **Albumentations Augmentations:** Handles advanced nadir spatial transformations (`RandomRotate90` & `Transpose` are added to exploit orientation invariance without introducing black borders), environmental simulations (`RandomShadow` & `ColorJitter` simulate different sun angles and shadows), focus blur (`Sharpen` & `Blur`), sensor noise (`GaussNoise` & `ISONoise`), and structural dropout.
* **Warping Restriction:** Perspective warp (`albu_warp_p`) is kept very low (max `0.1`) to prevent tree crowns from stretching into unrealistic, elongated shapes.
* **Important:** Any additional augmentations should be added on the **Albumentations side** rather than YOLO to keep the search space clean and decouple training parameters.

---

### 📂 Model-Specific Sweep Files

Since YOLO and Transformer-based RT-DETR models have different architectural constraints (e.g., RT-DETR does not use YOLO's native multi-image stitching and uses AdamW-specific learning schedules), the HPO configurations are separated:

#### 1. YOLO Sweep Configurations
* **`sweep_aug_yolo.yaml`** (Stage 1: Augmentation): Optimizes both YOLO native augmentations (`mosaic`, `mixup`, `copy_paste`) and Albumentations transformations.
  - *Initialize:* `wandb sweep sweep_aug_yolo.yaml`
* **`sweep_hpo_yolo.yaml`** (Stage 2: Learning Parameters HPO): Optimizes learning rates (`lr0`/`lrf`), momentum, weight decay, warmup epochs, and optimizers (SGD/AdamW).
  - *Initialize:* `wandb sweep sweep_hpo_yolo.yaml`

#### 2. RT-DETR Sweep Configurations
* **`sweep_aug_rtdetr.yaml`** (Stage 1: Augmentation): Optimizes only the Albumentations transformation pipeline, leaving YOLO native parameters inactive.
  - *Initialize:* `wandb sweep sweep_aug_rtdetr.yaml`
* **`sweep_hpo_rtdetr.yaml`** (Stage 2: Learning Parameters HPO): Optimizes learning rate (`lr0`/`lrf`), weight decay, warmup epochs, and optimizers (Adam/AdamW).
  - *Initialize:* `wandb sweep sweep_hpo_rtdetr.yaml`

---

### ⚙️ Executing a Sweep

#### Step A: Run Stage 1 (Augmentation Sweep)
1. Initialize the sweep (replace with the yaml configuration for your chosen model type):
   ```bash
   wandb sweep sweep_aug_yolo.yaml
   ```
2. Note the generated **Stage 1 Sweep ID** (e.g., `abc123xyz`).
3. Start local sweep agents to execute training:
   ```bash
   wandb agent <entity>/<project>/abc123xyz
   ```

#### Step B: Run Stage 2 (HPO Sweep)
1. Open the HPO config file matching your model (e.g. `sweep_hpo_yolo.yaml`) and update the `prev_aug_sweep_id` to your Stage 1 Sweep ID:
   ```yaml
   prev_aug_sweep_id:
     value: "abc123xyz"
   ```
2. Register and start agents:
   ```bash
   wandb sweep sweep_hpo_yolo.yaml
   wandb agent <entity>/<project>/<hpo_sweep_id>
   ```

---

## 🔄 Training, Evaluation, and Plotting Workflow

Here is how the complete batch execution, evaluation, and plotting cycle coordinates:

### 1. Automatic Evaluation
* **Does each training run get evaluated automatically?**
  * **Yes.** Right after a training run completes, the training script automatically calls `evaluate_model_coco()` from `eval_utils.py` to evaluate the checkpoint on the test split using strict `pycocotools` metrics.
  * This exports the final COCO metrics to a unique JSON file matching `evaluation_results/{run_name}_coco_metrics.json`.
  * For example, a run named `yolo26s_seed_42_best_aug` creates `evaluation_results/yolo26s_seed_42_best_aug_coco_metrics.json`.

### 2. Running and Comparing Multiple Seeds with Sweeps
* **How can I evaluate three runs with the same HPO & Augm settings and then plot the results?**
  * Simply execute a training run specifying both sweep IDs and a target model (omit the `--seed` flag so it automatically iterates through all configured seeds, e.g., 42, 100, and 999):
    ```bash
    pixi run train --model yolo26s.pt --aug-sweep-id <aug_id> --hpo-sweep-id <hpo_id>
    ```
  * This automatically launches 3 training runs:
    * `yolo26s_seed_42_best_aug_hpo`
    * `yolo26s_seed_100_best_aug_hpo`
    * `yolo26s_seed_999_best_aug_hpo`
  * Each run is trained with the exact same sweep hyperparameters and is evaluated automatically on completion, creating three distinct `{run_name}_coco_metrics.json` files.

### 3. Automatic Plotting and Seed Aggregation
* **How does plotting combine these results?**
  * When you run the plotting tool:
    ```bash
    pixi run plot
    ```
  * The plotting script reads all `*_coco_metrics.json` files.
  * It splits each run name (e.g., `yolo26s_seed_42_best_aug_hpo`) to separate the seed (`42`) and the sweep suffix (`_best_aug_hpo`).
  * It groups the files by the model configuration base (`yolo26s_best_aug_hpo`) and calculates the **Mean ± Standard Deviation** across the seeds.
  * Finally, it updates the summary markdown table and draws bar charts comparing the performance of different model configurations side-by-side.

### 4. Batch Execution
* **If I do a batch training/go through the whole suite, does this happen automatically?**
  * **Yes.** If you run `pixi run train` (or with sweep parameters but without specifying a single model/seed), it runs through every configured model/seed sequentially. Every single run performs its training and COCO test evaluation automatically, saving the metric files. After the batch completes, you only need to run `pixi run plot` once to aggregate and visualize everything.

### 5. Managing Multiple Suites & Clean Slates
* **Comparative Multi-Suite Support**: If you run multiple suites with different configurations (e.g., baseline first, then `_best_aug`), they generate distinct files due to the dynamic suffixes. Running `pixi run plot` aggregates all of them, showing side-by-side bar comparisons for the different configurations (e.g. `yolo26s` vs `yolo26s_best_aug`).
* **Overwriting Runs**: Re-running the exact same configuration and seed will overwrite the checkpoint directory and replace the corresponding `{run_name}_coco_metrics.json` file.
* **Resetting for a Clean Slate**: Since old metric JSONs persist in `evaluation_results/`, the plotter will continue to include them. If you change your dataset or want a fresh start, manually delete or archive the contents of:
  * `runs/` (Checkpoints)
  * `evaluation_results/` (Evaluation metrics and plots)
  * `wandb/` (Local W&B agent logs)

### 6. Step-by-Step Workflow Guide (Baseline vs. Sweep Comparative Study)

Follow these exact steps to run a baseline, optimize hyperparameters/augmentations, and run the optimized training suite to get comparative plots:

#### Step 1: Clean the Workspace (Optional)
If you want to start a fresh experiment (unrelated to past runs), clear the logging directories to prevent old configurations from showing up on the plots:
* Delete or move files out of `runs/` and `evaluation_results/`.

#### Step 2: Run the Baseline training suite
Train all configured models over the three default seeds under baseline/stock settings:
```bash
pixi run train
```
* **What happens:** This automatically trains the models and runs the COCO test evaluations, saving baseline JSON files like `yolo26s_seed_42_coco_metrics.json` inside the `evaluation_results/` folder.

#### Step 3: Plot the Baseline Results
```bash
pixi run plot
```
* **What happens:** Aggregates baseline runs across seeds and saves baseline comparative charts.

#### Step 4: Run HPO and Augmentation sweeps
Execute sweeps in W&B to find the best hyperparameters (using YOLO as an example):
1. **Initialize Phase 1 (Augmentation Sweep):**
   ```bash
   wandb sweep sweep_aug_yolo.yaml
   ```
   *Note the generated Stage 1 Sweep ID (e.g. `aug123`).*
2. **Start Phase 1 agents to optimize:**
   ```bash
   wandb agent <entity>/<project>/aug123
   ```
3. **Initialize Phase 2 (HPO Sweep):**
   * Open `sweep_hpo_yolo.yaml` and set `prev_aug_sweep_id` to `"aug123"`.
   * Start the sweep:
     ```bash
     wandb sweep sweep_hpo_yolo.yaml
     ```
     *Note the generated Stage 2 Sweep ID (e.g. `hpo456`).*
4. **Start Phase 2 agents to optimize:**
   ```bash
   wandb agent <entity>/<project>/hpo456
   ```

#### Step 5: Run training suite again with sweep parameters plugged in
**DO NOT delete or move the baseline logs in `evaluation_results/`.** Leaving them in place is required so the plotter can access both sets of results and compare them.
Run the training command passing both sweep IDs (and omit the `--seed` flag so it runs over all seeds):
```bash
pixi run train --aug-sweep-id aug123 --hpo-sweep-id hpo456
```
* **What happens:** Trains the configured models using the optimized augmentations and learning settings over all seeds, saving files like `yolo26s_seed_42_best_aug_hpo_coco_metrics.json`.

#### Step 6: Generate the comparative plots
```bash
pixi run plot
```
* **What happens:** The plotter loads **both** the baseline metric files and the sweep-optimized metric files. It aggregates the seeds for each configuration separately and generates side-by-side comparison bars showing the exact performance uplift (e.g. comparing `yolo26s` vs `yolo26s_best_aug_hpo`).

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
Sets up the command-line argument parser, defining options to override default models, seeds, epochs, batches, image sizes, devices, dataloader workers count, dataset fraction, runs directory, wandb directory, and sweep IDs (`--aug-sweep-id` and `--hpo-sweep-id`).

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
Sets up CLI arguments for standalone evaluation (accepts the same overrides as the training script, including `--aug-sweep-id` and `--hpo-sweep-id`).

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
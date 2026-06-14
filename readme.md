# Multi-Stage YOLO Hyperparameter Optimization Pipeline with W&B

This repository contains a two-stage hyperparameter optimization (HPO) pipeline for Ultralytics YOLO models using Weights & Biases (W&B) sweeps and Albumentations.

By separating the search space into an **Augmentation Phase (Stage 1)** and a **Learning Parameter Phase (Stage 2)**, we drastically reduce compute time and avoid a combinatorial explosion.

---

## 📋 Table of Contents

1. [Prerequisites & Setup](https://www.google.com/search?q=%23-prerequisites--setup)
2. [Configuration Single Source of Truth (`config.py`)](https://www.google.com/search?q=%231-configuration-single-source-of-truth)
3. [Stage 1: Augmentation Sweep](https://www.google.com/search?q=%23-stage-1-augmentation-sweep)
4. [Stage 2: Learning Parameters (HPO) Sweep](https://www.google.com/search?q=%23-stage-2-learning-parameters-hpo-sweep)
5. [Stage 3: Deploying to Production](https://www.google.com/search?q=%23-stage-3-deploying-to-production)

---

## 🔧 Prerequisites & Setup

Ensure you have the required packages installed and are logged into your Weights & Biases account:

```bash
pip install ultralytics wandb albumentations
wandb login

```

### File Structure

Your project directory should look like this:

```text
├── config.py          # Centralized configuration paths and helpers
├── run_sweep.py       # Unified execution script for both sweeps
├── sweep_aug.yaml     # Phase 1 configuration (Augmentations)
└── sweep_hpo.yaml     # Phase 2 configuration (Learning Parameters)

```

---

## 🛠️ Step-by-Step Execution Guide

### 1. Configure Your Environment (`config.py`)

Open `config.py` and modify the following variables inside the `PipelineConfig` class to match your setup:

* **`entity`**: Your W&B username or team name (e.g., `"brezemil"`).
* **`project`**: The name of your W&B project (e.g., `"24jun_batch_200_v2_geostrat"`).
* **`dataset_path`**: The absolute local path to your dataset's `dataset.yaml` file.
* **`model_variant`**: The starting YOLO base weights (e.g., `"yolo26s.pt"`).

---

### 2. Run Stage 1: Augmentation Sweep

In this stage, learning parameters are locked to recommended baseline values while W&B searches for the optimal geometric, warping, and color augmentations.

#### Step A: Initialize the Sweep

Run the following command in your terminal to register the sweep with the W&B server:

```bash
wandb sweep sweep_aug.yaml

```

#### Step B: Capture the Sweep ID

The terminal output will display a unique string looking like this:

```text
wandb: Creating sweep from: sweep_aug.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/brezemil/24jun_batch_200_v2_geostrat/sweeps/abc123xyz

```

👉 **Copy the generated 9-character ID (`abc123xyz`).** This is your **Stage 1 Sweep ID**.

#### Step C: Start the Sweep Agents

Launch one or more local agents to execute the training runs:

```bash
wandb agent brezemil/24jun_batch_200_v2_geostrat/abc123xyz

```

*(Replace `brezemil`, `24jun_batch_200_v2_geostrat`, and `abc123xyz` with your actual W&B entity, project name, and Stage 1 Sweep ID).*

---

### 3. Run Stage 2: Learning Parameters (HPO) Sweep

Once Stage 1 finishes, you will lock down the best discovered augmentation parameters and sweep across learning parameters like learning rates, momentum, and optimizers.

#### Step A: Link Stage 1 to Stage 2

Open your **`sweep_hpo.yaml`** file and find the `prev_aug_sweep_id` parameter. Paste your Stage 1 Sweep ID into the `value` field:

```yaml
parameters:
  phase:
    value: "hpo"
  prev_aug_sweep_id:
    value: "abc123xyz"  # 👈 PASTE YOUR STAGE 1 SWEEP ID HERE

```

#### Step B: Initialize the Phase 2 Sweep

Run the following command to register the second sweep config:

```bash
wandb sweep sweep_hpo.yaml

```

#### Step C: Capture the New Sweep ID

W&B will return a brand new unique ID for Stage 2:

```text
wandb: Created sweep with ID: hpo456def

```

👉 **Copy this new ID (`hpo456def`).** This is your **Stage 2 Sweep ID**.

#### Step D: Start Phase 2 Sweep Agents

Launch your agents to execute the HPO training runs:

```bash
wandb agent brezemil/24jun_batch_200_v2_geostrat/hpo456def

```

---

### 4. Production Deployment

After completing both sweeps, you are ready to train your final, full-length production model combining the best of both worlds.

To run a static, standalone production run without launching a sweep agent, you can extract your optimal parameters cleanly by using the built-in API helper in Python or running a manual override command:

```python
from config import PipelineConfig

# Fetch the ultimate configuration maps automatically
best_augs = PipelineConfig.get_best_sweep_config("abc123xyz", "24jun_batch_200_v2_geostrat", "brezemil")
best_hpo = PipelineConfig.get_best_sweep_config("hpo456def", "24jun_batch_200_v2_geostrat", "brezemil")

```

---

## 💡 Troubleshooting & Notes

> 📌 **VRAM Out of Memory (OOM):** If your GPU runs out of VRAM processing heavy $1024 \times 1024$ images, open `config.py` and change `batch_size: int = -1` to a fixed lower value like `16` or `8`.
> 📌 **Kill Active Agents:** To stop a sweep agent early, press `Ctrl + C` in the terminal window. The agent will finish its current epoch safely before exiting.

---

Here is a clean, structured Markdown section that you can copy and paste directly into your `README.md`. It uses clear tables and diagrams to explain the architecture to anyone reading your repository.

---

```markdown
## 🔄 Configuration Architecture & Data Flow

This pipeline uses a split architecture: **W&B Sweeps (YAML)** define the theoretical search space, while the **Execution Script (`run_sweep.py`)** handles the local runtime implementation. 

### The Data Lifecycle

1. **The Blueprint (Server-Side):** The W&B Cloud Server reads your `.yaml` file to understand parameter boundaries and distributions, using its Bayesian optimization model to determine the next combination to test.
2. **The Handshake:** Your local `wandb agent` fetches that single specific combination (e.g., `{"albu_spatial_p": 0.42, "phase": "augmentation"}`) via `wandb.init()`.
3. **The Translation (Local-Side):** The `run_sweep.py` script takes those raw numbers and dynamically builds active Python objects (like Albumentations pipelines) and updates the Ultralytics training dictionary.

---

## 🛠️ Modifying the Search Space (Cheat Sheet)

When expanding or altering your experiments, use this guide to determine whether you need to update the YAML configuration, the Python script, or both.

| Modification Goal | Change YAML? | Change Python? | Why? |
| :--- | :---: | :---: | :--- |
| **Adjust Limits** <br>*(e.g., Changing `mixup` max from 0.3 to 0.5)* | **Yes** | **No** | The Python script dynamically ingests whatever value the W&B server sends. |
| **Change Algorithms** <br>*(e.g., Switching from `bayes` to `random` search)* | **Yes** | **No** | Search strategies are managed entirely on the W&B server side. |
| **Add New Native YOLO Params** <br>*(e.g., Sweeping `label_smoothing`)* | **Yes** | **No** | The script uses dictionary unpacking (`**hpo_params`) to automatically forward new keys to `model.train()`. |
| **Add New Albumentations Effects** <br>*(e.g., Adding `A.RandomBrightnessContrast`)* | **Yes** | **Yes** | **YAML** must generate the probability variable, and **Python** must explicitly instantiate the new Albumentations class. |
| **Modify Hardcoded Baseline Rules** <br>*(e.g., Changing the locked Stage 1 learning rate)* | **No** | **Yes** | These are frozen pipeline rules handled entirely within the execution logic routing. |

### How to add a new Albumentations effect:

If you want to add a new image manipulation step to the sweep, follow these two steps:

1. **Update your YAML file** to register the new hyperparameter probability distribution:
   ```yaml
   parameters:
     albu_brightness_p: {distribution: uniform, min: 0.0, max: 0.5}

```

2. **Update `run_sweep.py**` inside the `build_albumentations_pipeline` function to map that parameter to the active pipeline:
```python
# Inside build_albumentations_pipeline:
A.RandomBrightnessContrast(p=config_dict.get("albu_brightness_p", 0.0))

```



```

```
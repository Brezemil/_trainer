"""
Multi-Stage YOLO Hyperparameter Optimization (HPO) Sweep Runner.

This script coordinates both Phase 1 (Augmentation tuning) and Phase 2 (Learning parameter HPO)
runs using Weights & Biases (W&B) and Albumentations. The execution flow is driven dynamically
by parameters injected via the W&B sweep configuration:

- Phase 1 ("augmentation"): Tunes native YOLO augmentations (mosaic, mixup, copy_paste)
  alongside an Albumentations pipeline with locked learning parameters.
- Phase 2 ("hpo"): Retrieves the best augmentation settings from a finished Phase 1 sweep,
  instantiates that Albumentations pipeline, and searches over learning rate, momentum,
  optimizers, and weight decay.
"""

from typing import List, Dict, Any
import wandb
import albumentations as A
from ultralytics import YOLO, settings
from config import PipelineConfig
import os
import warnings

#Suppress Python warnings (silences the torchvision warning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchvision.*")

#Suppress the pynvml/PyTorch internal FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Optional: Mute PyTorch hardware warning logs in subprocesses
os.environ["PYTHONWARNINGS"] = "ignore"

from eval_utils import build_albumentations_pipeline


def main() -> None:
    """
    Main entry point for running a sweep run agent.

    1. Loads global configuration settings from `PipelineConfig`.
    2. Initializes/subscribes to a W&B sweep agent run.
    3. Resets and disables standard native YOLO spatial and color augmentations
       to ensure custom Albumentations probabilities control the training environment.
    4. Evaluates the current sweep phase ('augmentation' or 'hpo') and executes the corresponding logic.
    5. Configures training parameters, runs the YOLO model training, and tags top-performing runs.
    """
    cfg = PipelineConfig()
    
    # Enable W&B integration in Ultralytics settings
    settings.update({"wandb": True})
    
    # Initialize the run (W&B agent automatically injects current sweep parameters into run.config)
    run = wandb.init(project=cfg.project, entity=cfg.entity)
    wb_config = dict(run.config)
    phase = wb_config.get("phase", "production")

    # Define baseline training parameters shared across all phases
    train_kwargs = {
        "data": cfg.dataset_path,
        "epochs": cfg.sweep_epochs,
        "imgsz": cfg.image_size,
        "device": cfg.device,
        "batch": cfg.batch_size,
        "workers": cfg.workers,
        "fraction": cfg.fraction,
        "exist_ok": True,
        "amp": cfg.amp,
        **cfg.fixed_loss
    }

    # Disable default YOLO spatial/color augmentations.
    # This ensures that our custom Albumentations transforms are fully isolated and evaluated
    # without overlapping influence from standard YOLO augmentation processes.
    train_kwargs.update({
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "scale": 0.0, "translate": 0.0, "shear": 0.0,
        "bgr": 0.0, "close_mosaic": 15
    })

    # Routing based on phase parameter
    if phase == "augmentation":
        # Phase 1: Sweeping augmentations. Keep learning hyperparameters locked to baseline values.
        train_kwargs.update({
            "optimizer": "MuSGD", "lr0": 0.0054, "lrf": 0.0495,
            "momentum": 0.947, "weight_decay": 0.00064, "warmup_epochs": 0.98
        })
        # Extract native YOLO augmentations from the W&B configuration parameters
        native_augs = {k: v for k, v in wb_config.items() if not k.startswith("albu_") and k != "phase"}
        train_kwargs.update(native_augs)
        
        # Inject the dynamically constructed Albumentations pipeline
        train_kwargs["augmentations"] = build_albumentations_pipeline(wb_config, cfg.image_size)

    elif phase == "hpo":
        # Phase 2: Hyperparameter optimization. Retrieve the best augmentations discovered in Phase 1.
        best_aug_config = cfg.get_best_sweep_config(wb_config["prev_aug_sweep_id"], cfg.project, cfg.entity)
        
        native_augs = {
            "mosaic": best_aug_config.get("mosaic", 0.0), 
            "mixup": best_aug_config.get("mixup", 0.0), 
            "copy_paste": best_aug_config.get("copy_paste", 0.0)
        }
        
        # Isolate learning rate/optimizer parameters from the current run config
        hpo_params = {k: v for k, v in wb_config.items() if k not in ["phase", "prev_aug_sweep_id"]}
        
        train_kwargs.update(native_augs)
        train_kwargs.update(hpo_params)
        
        # Build Albumentations pipeline utilizing optimal probabilities from Phase 1
        train_kwargs["augmentations"] = build_albumentations_pipeline(best_aug_config, cfg.image_size)

    # Initialize model and begin training. Use RTDETR class for RT-DETR models.
    if "rtdetr" in cfg.model_variant.lower():
        from ultralytics import RTDETR
        print(f"Loading RT-DETR model: {cfg.model_variant}")
        model = RTDETR(cfg.model_variant)
    else:
        print(f"Loading YOLO model: {cfg.model_variant}")
        model = YOLO(cfg.model_variant)
    model.train(**train_kwargs)
    
    # Automatically tag top-performing runs on the W&B dashboard for easy filtering
    if run.summary.get("metrics/mAP50-95(B)", 0) > 0.5:
        run.tags = run.tags + ("top_performer",) if run.tags else ("top_performer",)
        
    # Close W&B run cleanly
    wandb.finish()


if __name__ == "__main__":
    main()
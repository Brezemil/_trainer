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


def build_albumentations_pipeline(config_dict: Dict[str, Any], imgsz: int) -> List[Any]:
    """
    Dynamically constructs an Albumentations augmentation pipeline based on W&B sweep parameters.

    Individual augmentations are parameterized with execution probabilities ('p') supplied 
    via the configuration dictionary. If a parameter is not present in the sweep dictionary, 
    the probability defaults to 0.0 (deactivated).

    Args:
        config_dict (Dict[str, Any]): Dictionary containing hyperparameters (e.g., probability limits).
        imgsz (int): Target image width and height for spatial resizing crop operations.

    Returns:
        List[Any]: A list of instantiated Albumentations transformation objects ready to be 
                   passed to the Ultralytics training pipeline.
    """
    return [
        # Spatial transformations
        A.RandomRotate90(p=config_dict.get("albu_spatial_p", 0.0)),
        A.HorizontalFlip(p=config_dict.get("albu_spatial_p", 0.0)),
        A.VerticalFlip(p=config_dict.get("albu_spatial_p", 0.0)),
        A.Transpose(p=config_dict.get("albu_spatial_p", 0.0)),
        
        # Perspective & Warp transformations
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0), 
            A.Affine(shear=(-5, 5), p=1.0), 
        ], p=config_dict.get("albu_warp_p", 0.0)),
        
        # Resizing and cropping
        A.RandomResizedCrop(
            size=(imgsz, imgsz), scale=(0.4, 1.0), ratio=(0.9, 1.1), p=config_dict.get("albu_crop_p", 0.0)
        ),
        
        # Texture & Blur
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=config_dict.get("albu_texture_p", 0.0)),
        
        # Color adjustments
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.ToGray(p=0.1), 
            A.Solarize(threshold=128, p=0.05), 
        ], p=config_dict.get("albu_color_p", 0.0)),
        
        # Coarse dropout (hole masking)
        A.CoarseDropout(
            num_holes_range=(8, 12), hole_height_range=(0.02, 0.05), 
            hole_width_range=(0.02, 0.05), p=config_dict.get("albu_dropout_p", 0.0)
        ),
        
        # Noise & Compression artifacts
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.ImageCompression(quality_range=(75, 100), p=1.0),
        ], p=config_dict.get("albu_noise_p", 0.0)),
    ]


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
        "exist_ok": True,
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

    # Initialize model and begin training
    model = YOLO(cfg.model_variant)
    model.train(**train_kwargs)
    
    # Automatically tag top-performing runs on the W&B dashboard for easy filtering
    if run.summary.get("metrics/mAP50-95(B)", 0) > 0.5:
        run.tags = run.tags + ("top_performer",) if run.tags else ("top_performer",)
        
    # Close W&B run cleanly
    wandb.finish()


if __name__ == "__main__":
    main()
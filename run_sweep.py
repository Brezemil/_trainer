import wandb
import albumentations as A
from ultralytics import YOLO, settings
from config import PipelineConfig

def build_albumentations_pipeline(config_dict, imgsz):
    """Dynamically builds Pipeline A based on provided W&B parameters."""
    return [
        A.RandomRotate90(p=config_dict.get("albu_spatial_p", 0.0)),
        A.HorizontalFlip(p=config_dict.get("albu_spatial_p", 0.0)),
        A.VerticalFlip(p=config_dict.get("albu_spatial_p", 0.0)),
        A.Transpose(p=config_dict.get("albu_spatial_p", 0.0)),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0), 
            A.Affine(shear=(-5, 5), p=1.0), 
        ], p=config_dict.get("albu_warp_p", 0.0)),
        A.RandomResizedCrop(
            size=(imgsz, imgsz), scale=(0.4, 1.0), ratio=(0.9, 1.1), p=config_dict.get("albu_crop_p", 0.0)
        ),
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=config_dict.get("albu_texture_p", 0.0)),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.ToGray(p=0.1), 
            A.Solarize(threshold=128, p=0.05), 
        ], p=config_dict.get("albu_color_p", 0.0)),
        A.CoarseDropout(
            num_holes_range=(8, 12), hole_height_range=(0.02, 0.05), 
            hole_width_range=(0.02, 0.05), p=config_dict.get("albu_dropout_p", 0.0)
        ),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.ImageCompression(quality_range=(75, 100), p=1.0),
        ], p=config_dict.get("albu_noise_p", 0.0)),
    ]

def main():
    cfg = PipelineConfig()
    settings.update({"wandb": True})
    
    # Initialize W&B run; it will automatically load config from the YAML sweep agent
    run = wandb.init(project=cfg.project, entity=cfg.entity)
    wb_config = dict(run.config)
    phase = wb_config.get("phase", "production")

    # Construct baseline parameters
    train_kwargs = {
        "data": cfg.dataset_path,
        "epochs": cfg.sweep_epochs,
        "imgsz": cfg.image_size,
        "batch": cfg.batch_size,
        "exist_ok": True,
        **cfg.fixed_loss
    }

    # Disable default YOLO spatial/color overlaps to enforce Albumentations isolation
    train_kwargs.update({
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "scale": 0.0, "translate": 0.0, "shear": 0.0,
        "bgr": 0.0, "close_mosaic": 15
    })

    # Phase Logic Routing
    if phase == "augmentation":
        # Sweeping augmentations, lock HPO variables
        train_kwargs.update({
            "optimizer": "MuSGD", "lr0": 0.0054, "lrf": 0.0495,
            "momentum": 0.947, "weight_decay": 0.00064, "warmup_epochs": 0.98
        })
        # Extract native augs vs albumentations
        native_augs = {k: v for k, v in wb_config.items() if not k.startswith("albu_") and k != "phase"}
        train_kwargs.update(native_augs)
        train_kwargs["augmentations"] = build_albumentations_pipeline(wb_config, cfg.image_size)

    elif phase == "hpo":
        # Sweeping HPO, fetch best augmentations from Phase 1
        best_aug_config = cfg.get_best_sweep_config(wb_config["prev_aug_sweep_id"], cfg.project, cfg.entity)
        
        native_augs = {"mosaic": best_aug_config.get("mosaic", 0.0), "mixup": best_aug_config.get("mixup", 0.0), "copy_paste": best_aug_config.get("copy_paste", 0.0)}
        hpo_params = {k: v for k, v in wb_config.items() if k not in ["phase", "prev_aug_sweep_id"]}
        
        train_kwargs.update(native_augs)
        train_kwargs.update(hpo_params)
        train_kwargs["augmentations"] = build_albumentations_pipeline(best_aug_config, cfg.image_size)

    # Execute Training
    model = YOLO(cfg.model_variant)
    model.train(**train_kwargs)
    
    # Tag best runs automatically based on validation metric
    if run.summary.get("metrics/mAP50-95(B)", 0) > 0.5:
        run.tags = run.tags + ("top_performer",) if run.tags else ("top_performer",)
        
    wandb.finish()

if __name__ == "__main__":
    main()
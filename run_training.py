"""
Stock Training Run Execution Script.

This script executes training runs for YOLO11, YOLO26, and RT-DETR at stock settings.
By default, it trains all models specified in PipelineConfig across all configured seeds.
It also supports running a single model or seed via command-line arguments.
Each run is tracked on Weights & Biases (W&B) with a sensible naming convention.
"""

import argparse
import sys
from typing import List
import wandb
from ultralytics import YOLO, settings
from config import PipelineConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11, YOLO26, and RT-DETR at stock settings.")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Specify a single model to train (e.g. yolo11s.pt, yolo26s.pt, rtdetr-l.pt). If not specified, trains all configured models."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Specify a single seed to train with. If not specified, trains across all configured seeds."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Override the number of epochs to train for."
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=None, 
        help="Override the batch size."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Override the device (e.g., 0 or cpu)."
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=None, 
        help="Override the image size."
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help="Override the number of dataloader workers."
    )
    parser.add_argument(
        "--fraction", 
        type=float, 
        default=None, 
        help="Override the fraction of dataset to train on (e.g. 0.01 for 1%% of data)."
    )
    parser.add_argument(
        "--runs-dir", 
        type=str, 
        default=None, 
        help="Override the runs directory for training checkpoints."
    )
    parser.add_argument(
        "--wandb-dir", 
        type=str, 
        default=None, 
        help="Override the wandb log directory."
    )
    parser.add_argument(
        "--aug-sweep-id", 
        type=str, 
        default=None, 
        help="W&B sweep ID for Phase 1 (augmentation tuning)."
    )
    parser.add_argument(
        "--hpo-sweep-id", 
        type=str, 
        default=None, 
        help="W&B sweep ID for Phase 2 (HPO)."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Load configuration from the single source of truth
    cfg = PipelineConfig()
    
    # Enable W&B integration in Ultralytics settings
    settings.update({"wandb": True})
    
    # Determine which models to train
    if args.model:
        models_to_train = [args.model]
    else:
        # Convert tuple/list from config to a list
        models_to_train = list(cfg.models)
        
    # Determine which seeds to use
    if args.seed is not None:
        seeds_to_use = [args.seed]
    else:
        seeds_to_use = list(cfg.seeds)
        
    epochs = args.epochs if args.epochs is not None else cfg.prod_epochs
    batch_size = args.batch if args.batch is not None else cfg.batch_size
    device = args.device if args.device is not None else cfg.device
    imgsz = args.imgsz if args.imgsz is not None else cfg.image_size
    workers = args.workers if args.workers is not None else cfg.workers
    fraction = args.fraction if args.fraction is not None else cfg.fraction
    runs_dir = args.runs_dir if args.runs_dir is not None else cfg.runs_dir
    wandb_dir = args.wandb_dir if args.wandb_dir is not None else cfg.wandb_dir
    
    aug_sweep_id = args.aug_sweep_id if args.aug_sweep_id is not None else cfg.aug_sweep_id
    hpo_sweep_id = args.hpo_sweep_id if args.hpo_sweep_id is not None else cfg.hpo_sweep_id
    
    # Resolve relative paths to absolute paths
    import os
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.abspath(runs_dir)
    if not os.path.isabs(wandb_dir):
        wandb_dir = os.path.abspath(wandb_dir)
    
    # Fetch best sweep configurations if sweep IDs are provided
    best_aug_config = None
    if aug_sweep_id:
        print(f"\nFetching best configuration from augmentation sweep: {aug_sweep_id}...")
        try:
            best_aug_config = cfg.get_best_sweep_config(aug_sweep_id, cfg.project, cfg.entity)
            print("Successfully loaded best augmentation configurations.")
        except Exception as e:
            print(f"Error fetching augmentation sweep config: {e}. Proceeding without sweep results.", file=sys.stderr)
            
    best_hpo_config = None
    if hpo_sweep_id:
        print(f"\nFetching best configuration from HPO sweep: {hpo_sweep_id}...")
        try:
            best_hpo_config = cfg.get_best_sweep_config(hpo_sweep_id, cfg.project, cfg.entity)
            print("Successfully loaded best HPO configurations.")
        except Exception as e:
            print(f"Error fetching HPO sweep config: {e}. Proceeding without sweep results.", file=sys.stderr)
            
    print("=" * 60)
    print("Starting stock training runs configuration:")
    print(f"Models: {models_to_train}")
    print(f"Seeds: {seeds_to_use}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Workers: {workers}")
    print(f"Dataset Fraction: {fraction}")
    print(f"Runs Dir: {runs_dir}")
    print(f"W&B Dir: {wandb_dir}")
    print(f"Dataset: {cfg.dataset_path}")
    print(f"W&B Entity: {cfg.entity}")
    print(f"W&B Project: {cfg.project}")
    print(f"Aug Sweep ID: {aug_sweep_id}")
    print(f"HPO Sweep ID: {hpo_sweep_id}")
    print("=" * 60)
    
    total_runs = len(models_to_train) * len(seeds_to_use)
    run_idx = 1
    
    for model_name in models_to_train:
        for seed in seeds_to_use:
            model_base = model_name.replace(".pt", "")
            
            # Construct run name with suffixes indicating sweep integration
            suffix = ""
            if aug_sweep_id and hpo_sweep_id:
                suffix = "_best_aug_hpo"
            elif aug_sweep_id:
                suffix = "_best_aug"
            elif hpo_sweep_id:
                suffix = "_best_hpo"
                
            run_name = f"{model_base}_seed_{seed}{suffix}"
            
            print(f"\n[{run_idx}/{total_runs}] Initializing training for: {run_name}...")
            
            # Ensure wandb_dir exists
            os.makedirs(wandb_dir, exist_ok=True)
            
            # Prepare configuration dictionary for W&B
            wb_run_config = {
                "model": model_name,
                "seed": seed,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch_size": batch_size,
                "device": device,
                "workers": workers,
                "fraction": fraction,
                "runs_dir": runs_dir,
                "wandb_dir": wandb_dir,
                "dataset": cfg.dataset_path,
            }
            if aug_sweep_id:
                wb_run_config["aug_sweep_id"] = aug_sweep_id
                if best_aug_config:
                    for k, v in best_aug_config.items():
                        if k.startswith("albu_") or k in ["mosaic", "mixup", "copy_paste"]:
                            wb_run_config[f"best_aug_{k}"] = v
            if hpo_sweep_id:
                wb_run_config["hpo_sweep_id"] = hpo_sweep_id
                if best_hpo_config:
                    for k, v in best_hpo_config.items():
                        if k in ["optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"]:
                            wb_run_config[f"best_hpo_{k}"] = v
            
            # Initialize W&B run cleanly
            run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=run_name,
                dir=wandb_dir,
                config=wb_run_config,
                reinit=True
            )
            
            try:
                # Load the model variant. Use RTDETR class for RT-DETR models.
                is_rtdetr = "rtdetr" in model_name.lower()
                if is_rtdetr:
                    from ultralytics import RTDETR
                    print(f"Loading RT-DETR model: {model_name}")
                    model = RTDETR(model_name)
                else:
                    print(f"Loading YOLO model: {model_name}")
                    model = YOLO(model_name)
                
                # Assemble training arguments
                train_kwargs = {
                    "data": cfg.dataset_path,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "device": device,
                    "batch": batch_size,
                    "seed": seed,
                    "workers": workers,
                    "fraction": fraction,
                    "project": runs_dir,
                    "name": run_name,
                    "exist_ok": True,
                }
                
                # Apply fixed loss weights from baseline config to keep consistency
                if cfg.fixed_loss:
                    train_kwargs.update(cfg.fixed_loss)
                    
                if best_aug_config:
                    # Disable default YOLO spatial/color augmentations to isolate sweep influence
                    train_kwargs.update({
                        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                        "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0,
                        "scale": 0.0, "translate": 0.0, "shear": 0.0,
                        "bgr": 0.0, "close_mosaic": 15
                    })
                    
                    # Add native YOLO augmentations if they were swept and if this is a YOLO model
                    if not is_rtdetr:
                        native_augs = {
                            "mosaic": best_aug_config.get("mosaic", 0.0),
                            "mixup": best_aug_config.get("mixup", 0.0),
                            "copy_paste": best_aug_config.get("copy_paste", 0.0)
                        }
                        train_kwargs.update(native_augs)
                        
                    # Build and inject custom Albumentations pipeline
                    from eval_utils import build_albumentations_pipeline
                    train_kwargs["augmentations"] = build_albumentations_pipeline(best_aug_config, imgsz)
                    
                if best_hpo_config:
                    hpo_keys = ["optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"]
                    hpo_params = {k: best_hpo_config[k] for k in hpo_keys if k in best_hpo_config}
                    train_kwargs.update(hpo_params)
                    
                print("Final training arguments:")
                for k, v in train_kwargs.items():
                    if k == "augmentations":
                        print(f"  {k}: <custom Albumentations pipeline with {len(v)} stages>")
                    else:
                        print(f"  {k}: {v}")
                
                # Run the model training with assembled settings
                model.train(**train_kwargs)
                print(f"Successfully finished training run: {run_name}")
                
                # Evaluate on the test split using pycocotools
                from eval_utils import evaluate_model_coco
                evaluate_model_coco(
                    model_path_or_model=model,
                    dataset_yaml_path=cfg.dataset_path,
                    split="test",
                    eval_results_dir=cfg.eval_results_dir,
                    run_name=run_name,
                    device=device,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    workers=workers
                )
            except Exception as e:
                print(f"Error occurred during training of {run_name}: {e}", file=sys.stderr)
            finally:
                # Close the W&B run
                wandb.finish()
                run_idx += 1

    print("\nAll training runs complete!")

if __name__ == "__main__":
    main()

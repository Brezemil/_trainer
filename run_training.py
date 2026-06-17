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
    
    # Resolve relative paths to absolute paths
    import os
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.abspath(runs_dir)
    if not os.path.isabs(wandb_dir):
        wandb_dir = os.path.abspath(wandb_dir)
    
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
    print("=" * 60)
    
    total_runs = len(models_to_train) * len(seeds_to_use)
    run_idx = 1
    
    for model_name in models_to_train:
        for seed in seeds_to_use:
            model_base = model_name.replace(".pt", "")
            run_name = f"{model_base}_seed_{seed}"
            
            print(f"\n[{run_idx}/{total_runs}] Initializing training for: {run_name}...")
            
            # Ensure wandb_dir exists
            os.makedirs(wandb_dir, exist_ok=True)
            
            # Initialize W&B run cleanly
            run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=run_name,
                dir=wandb_dir,
                config={
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
                },
                reinit=True
            )
            
            try:
                # Load the model variant. Use RTDETR class for RT-DETR models.
                if "rtdetr" in model_name.lower():
                    from ultralytics import RTDETR
                    print(f"Loading RT-DETR model: {model_name}")
                    model = RTDETR(model_name)
                else:
                    print(f"Loading YOLO model: {model_name}")
                    model = YOLO(model_name)
                
                # Run the model training with stock settings, saving to runs_dir
                model.train(
                    data=cfg.dataset_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    device=device,
                    batch=batch_size,
                    seed=seed,
                    workers=workers,
                    fraction=fraction,
                    project=runs_dir,
                    name=run_name,
                    exist_ok=True,
                )
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

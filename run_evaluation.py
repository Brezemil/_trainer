"""
Standalone Evaluation Script.

This script runs strict COCO evaluation (via pycocotools) on the test split for trained checkpoints.
It checks for checkpoints at runs/{run_name}/weights/best.pt and evaluates them.
"""

import argparse
import os
import sys
from config import PipelineConfig
from eval_utils import evaluate_model_coco

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO/RT-DETR checkpoints on the test split.")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Specify a single model to evaluate (e.g. yolo11s.pt, yolo26s.pt, rtdetr-l.pt)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Specify a single seed to evaluate."
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        help="Dataset split to evaluate on (default: test)."
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=None, 
        help="Override batch size."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Override device (e.g., 0 or cpu)."
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=None, 
        help="Override image size."
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help="Override number of dataloader workers."
    )
    parser.add_argument(
        "--runs-dir", 
        type=str, 
        default=None, 
        help="Override the runs directory where training checkpoints are saved."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    cfg = PipelineConfig()
    
    # Resolve parameters
    models_to_eval = [args.model] if args.model else list(cfg.models)
    seeds_to_use = [args.seed] if args.seed is not None else list(cfg.seeds)
    
    split = args.split
    batch_size = args.batch if args.batch is not None else cfg.batch_size
    device = args.device if args.device is not None else cfg.device
    imgsz = args.imgsz if args.imgsz is not None else cfg.image_size
    workers = args.workers if args.workers is not None else cfg.workers
    runs_dir = args.runs_dir if args.runs_dir is not None else cfg.runs_dir
    
    # Resolve relative path to absolute
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.abspath(runs_dir)
    
    print("=" * 60)
    print(f"Starting COCO evaluation on {split} split:")
    print(f"Models: {models_to_eval}")
    print(f"Seeds: {seeds_to_use}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Workers: {workers}")
    print(f"Runs Directory: {runs_dir}")
    print(f"Dataset: {cfg.dataset_path}")
    print(f"Results Directory: {cfg.eval_results_dir}")
    print("=" * 60)
    
    evaluated_count = 0
    missing_count = 0
    
    for model_name in models_to_eval:
        for seed in seeds_to_use:
            model_base = model_name.replace(".pt", "")
            run_name = f"{model_base}_seed_{seed}"
            
            # Locate checkpoint
            checkpoint_path = os.path.join(runs_dir, run_name, "weights", "best.pt")
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found for {run_name} at: {checkpoint_path}. Skipping.")
                missing_count += 1
                continue
                
            print(f"\nEvaluating found checkpoint: {checkpoint_path}")
            try:
                evaluate_model_coco(
                    model_path_or_model=checkpoint_path,
                    dataset_yaml_path=cfg.dataset_path,
                    split=split,
                    eval_results_dir=cfg.eval_results_dir,
                    run_name=run_name,
                    device=device,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    workers=workers
                )
                evaluated_count += 1
            except Exception as e:
                print(f"Error evaluating {run_name}: {e}", file=sys.stderr)
                
    print(f"\nEvaluation complete. Evaluated: {evaluated_count}, Missing: {missing_count}")

if __name__ == "__main__":
    main()

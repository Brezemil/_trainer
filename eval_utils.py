import os
import json
import glob
import shutil
import sys
from typing import Dict, Any, List
from PIL import Image
import yaml

def parse_dataset_yaml(dataset_yaml_path: str) -> Dict[str, Any]:
    """Parses dataset.yaml to extract path details and class names."""
    with open(dataset_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data

def generate_coco_gt(dataset_yaml_path: str, split: str, save_path: str) -> str:
    """
    Converts a YOLO format dataset split annotations to a COCO ground-truth JSON file.
    If the file already exists, it skips generation to save time.
    """
    if os.path.exists(save_path):
        print(f"Ground truth COCO JSON already exists at {save_path}. Skipping generation.")
        return save_path

    print(f"Generating COCO ground truth for split: {split}...")
    db_info = parse_dataset_yaml(dataset_yaml_path)
    
    # Resolve absolute paths
    base_path = db_info.get("path", "")
    split_img_dir = db_info.get(split, "")
    
    # Handle absolute vs relative paths in yaml
    if not os.path.isabs(base_path):
        # If path in yaml is relative, resolve it relative to dataset.yaml's folder
        yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
        base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        
    images_dir = os.path.join(base_path, split_img_dir) if not os.path.isabs(split_img_dir) else split_img_dir
    
    # Labels dir is typically parallel to images dir (replace images with labels in path)
    # E.g. /path/to/images/test -> /path/to/labels/test
    labels_dir = images_dir.replace("images", "labels")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
    # Gather images
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    name_to_id = {}
    images_list = []
    
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(filename)
        name_to_id[img_name] = idx
        
        with Image.open(img_path) as img:
            w, h = img.size
            
        images_list.append({
            "id": idx,
            "file_name": filename,
            "width": w,
            "height": h
        })
        
    # Build categories list
    names_dict = db_info.get("names", {})
    categories = []
    for cls_id, cls_name in names_dict.items():
        # Map YOLO class_id (0-indexed) to COCO category_id (1-indexed)
        categories.append({
            "id": int(cls_id) + 1,
            "name": str(cls_name),
            "supercategory": "none"
        })
        
    # Build annotations
    annotations = []
    ann_id_counter = 1
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(filename)
        image_id = name_to_id[img_name]
        
        # Check label file
        label_path = os.path.join(labels_dir, img_name + ".txt")
        if not os.path.exists(label_path):
            continue
            
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            
            # Convert YOLO normalized center format to COCO absolute pixel format [x_min, y_min, w, h]
            w_pixel = w * img_w
            h_pixel = h * img_h
            x_min = (x_c - w / 2.0) * img_w
            y_min = (y_c - h / 2.0) * img_h
            area = w_pixel * h_pixel
            
            annotations.append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": cls_id + 1, # 1-indexed for COCO
                "bbox": [x_min, y_min, w_pixel, h_pixel],
                "area": area,
                "iscrowd": 0
            })
            ann_id_counter += 1
            
    coco_gt_dict = {
        "images": images_list,
        "annotations": annotations,
        "categories": categories
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(coco_gt_dict, f, indent=4)
        
    print(f"Successfully generated COCO ground truth at {save_path} with {len(images_list)} images.")
    return save_path

def evaluate_model_coco(
    model_path_or_model: Any,
    dataset_yaml_path: str,
    split: str,
    eval_results_dir: str,
    run_name: str,
    device: Any = 0,
    batch_size: int = 16,
    imgsz: int = 640,
    workers: int = 0
) -> Dict[str, Any]:
    """
    Runs model validation on the specified split using Ultralytics to export predictions to JSON,
    maps the image IDs to integer COCO format, runs pycocotools COCOeval, and saves the metrics.
    """
    from ultralytics import YOLO, RTDETR
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    eval_results_dir = os.path.abspath(eval_results_dir)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # 1. Load the model
    if isinstance(model_path_or_model, str):
        model_name = os.path.basename(model_path_or_model)
        if "rtdetr" in model_name.lower():
            model = RTDETR(model_path_or_model)
        else:
            model = YOLO(model_path_or_model)
    else:
        model = model_path_or_model
        
    print(f"Evaluating {run_name} on the {split} set...")
    
    # 2. Run validation to get predictions.json
    # We save into a temporary validation folder inside the eval_results_dir
    temp_dir_name = f"temp_val_{run_name}"
    val_results = model.val(
        data=dataset_yaml_path,
        split=split,
        save_json=True,
        device=device,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        project=eval_results_dir,
        name=temp_dir_name,
        exist_ok=True,
        plots=False  # speed up validation
    )
    
    temp_val_path = os.path.join(eval_results_dir, temp_dir_name)
    pred_src_file = os.path.join(temp_val_path, "predictions.json")
    
    if not os.path.exists(pred_src_file):
        raise FileNotFoundError(f"Ultralytics failed to save predictions.json at {pred_src_file}")
        
    # 3. Generate COCO ground-truth file
    gt_file = os.path.join(eval_results_dir, f"coco_gt_{split}.json")
    generate_coco_gt(dataset_yaml_path, split, gt_file)
    
    # 4. Load ground truth to map image filename string to int ID
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
        
    name_to_id = {}
    for img in gt_data["images"]:
        # Match base filename (without extension)
        img_name = os.path.splitext(img["file_name"])[0]
        name_to_id[img_name] = img["id"]
        
    # 5. Load and map predictions.json
    with open(pred_src_file, "r") as f:
        preds = json.load(f)
        
    mapped_preds = []
    unmapped_count = 0
    
    for p in preds:
        img_name = p["image_id"]
        # Strip path or extension if present in image_id
        img_name_clean = os.path.splitext(os.path.basename(img_name))[0]
        
        if img_name_clean in name_to_id:
            mapped_preds.append({
                "image_id": name_to_id[img_name_clean],
                "category_id": p["category_id"],
                "bbox": p["bbox"],
                "score": p["score"]
            })
        else:
            unmapped_count += 1
            
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} predictions could not be mapped to ground-truth image IDs.")
        
    # Save mapped predictions
    pred_file = os.path.join(eval_results_dir, f"{run_name}_coco_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(mapped_preds, f, indent=4)
        
    # 6. Run pycocotools COCOeval
    coco_gt = COCO(gt_file)
    
    if len(mapped_preds) == 0:
        print("Warning: No predictions found, creating dummy evaluator results.")
        stats = [0.0] * 12
    else:
        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = list(coco_eval.stats)
        
    # Format metrics dictionary
    metrics = {
        "model_variant": getattr(model, "ckpt_path", "") or str(model_path_or_model) if isinstance(model_path_or_model, str) else "unknown",
        "run_name": run_name,
        "split": split,
        "metrics": {
            "AP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "AP_small": stats[3],
            "AP_medium": stats[4],
            "AP_large": stats[5],
            "AR_max1": stats[6],
            "AR_max10": stats[7],
            "AR_max100": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11]
        }
    }
    
    # Save metrics JSON
    metrics_file = os.path.join(eval_results_dir, f"{run_name}_coco_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Strict COCO metrics saved to {metrics_file}")
    
    # 7. Clean up temporary directories
    try:
        shutil.rmtree(temp_val_path)
    except Exception as e:
        print(f"Error cleaning up temp directory {temp_val_path}: {e}")
        
    return metrics

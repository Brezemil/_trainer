"""
Results Aggregation and Plotting Script.

This script reads all metric files matching *_coco_metrics.json in the evaluation results folder,
groups them by model name, calculates mean and standard deviation across different seeds,
creates comparison plots, and saves a markdown summary table.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from config import PipelineConfig

def load_metrics(eval_dir: str) -> List[Dict[str, Any]]:
    """Loads all COCO metrics JSON files from the evaluation results directory."""
    pattern = os.path.join(eval_dir, "*_coco_metrics.json")
    files = glob.glob(pattern)
    
    results = []
    for filepath in files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    return results

def group_and_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Groups results by model variant and aggregates across seeds
    calculating mean and standard deviation.
    """
    grouped_data = {}
    
    for r in results:
        run_name = r["run_name"]
        
        # Parse model base name and seed from run_name
        # E.g. "yolo11s_seed_42" -> model_base="yolo11s", seed="42"
        # E.g. "yolo11s_seed_42_best_aug" -> model_base="yolo11s_best_aug", seed="42"
        if "_seed_" in run_name:
            parts = run_name.split("_seed_")
            model_prefix = parts[0]
            seed_and_suffix = parts[1].split("_", 1)
            seed_str = seed_and_suffix[0]
            if len(seed_and_suffix) > 1:
                model_base = f"{model_prefix}_{seed_and_suffix[1]}"
            else:
                model_base = model_prefix
        else:
            model_base = run_name
            
        if model_base not in grouped_data:
            grouped_data[model_base] = {metric: [] for metric in r["metrics"].keys()}
            
        for metric_name, val in r["metrics"].items():
            grouped_data[model_base][metric_name].append(val)
            
    # Calculate stats
    stats = {}
    for model_base, metrics in grouped_data.items():
        stats[model_base] = {}
        for metric_name, values in metrics.items():
            vals = np.array(values)
            stats[model_base][metric_name] = {
                "values": values,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                "count": len(vals)
            }
            
    return stats

def plot_ap_comparison(stats: Dict[str, Dict[str, Dict[str, float]]], save_dir: str) -> None:
    """Generates a bar plot of AP and AP50 with standard deviation error bars."""
    models = list(stats.keys())
    if not models:
        print("No models found to plot.")
        return
        
    ap_means = [stats[m]["AP"]["mean"] for m in models]
    ap_stds = [stats[m]["AP"]["std"] for m in models]
    
    ap50_means = [stats[m]["AP50"]["mean"] for m in models]
    ap50_stds = [stats[m]["AP50"]["std"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, ap_means, width, yerr=ap_stds, label='mAP@0.50:0.95', 
                    capsize=5, color='#4A90E2', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, ap50_means, width, yerr=ap50_stds, label='mAP@0.50', 
                    capsize=5, color='#50E3C2', edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Score')
    ax.set_title('Strict COCO AP Metrics Comparison (Mean ± Std across Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plot_path = os.path.join(save_dir, "ap_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved AP comparison plot to {plot_path}")

def plot_individual_seeds(stats: Dict[str, Dict[str, Dict[str, float]]], save_dir: str) -> None:
    """Generates a scatter/swarm plot showing individual seed scores for each model."""
    models = list(stats.keys())
    if not models:
        return
        
    x_data = []
    y_data = []
    
    for m in models:
        vals = stats[m]["AP"]["values"]
        for v in vals:
            x_data.append(m)
            y_data.append(v)
            
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot mean with a bar
    for i, m in enumerate(models):
        mean_val = stats[m]["AP"]["mean"]
        plt.bar(m, mean_val, color='gray', alpha=0.2, edgecolor='black', width=0.5)
        
    # Plot individual seeds
    sns.stripplot(x=x_data, y=y_data, hue=x_data, jitter=0.15, size=10, linewidth=1, palette="Set2", legend=False)
    
    plt.title('mAP@0.50:0.95 Scores by Seed')
    plt.ylabel('mAP@0.50:0.95')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(save_dir, "ap_seed_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved seed distribution plot to {plot_path}")

def generate_markdown_summary(stats: Dict[str, Dict[str, Dict[str, float]]], save_dir: str) -> None:
    """Generates a markdown table summarizing the evaluation results and saves it to a file."""
    models = sorted(list(stats.keys()))
    if not models:
        return
        
    lines = [
        "# Strict Test Set Evaluation Summary (COCO Metrics)",
        "",
        "This table aggregates the strict pycocotools AP/AR metrics computed on the test split.",
        "Scores are reported as **Mean ± Standard Deviation** across the configured seeds.",
        "",
        "| Model Variant | Runs | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | AP (Small) | AP (Medium) | AP (Large) | AR@100 |",
        "| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for m in models:
        count = stats[m]["AP"]["count"]
        ap = f"{stats[m]['AP']['mean']:.5f} ± {stats[m]['AP']['std']:.5f}"
        ap50 = f"{stats[m]['AP50']['mean']:.5f} ± {stats[m]['AP50']['std']:.5f}"
        ap75 = f"{stats[m]['AP75']['mean']:.5f} ± {stats[m]['AP75']['std']:.5f}"
        ap_s = f"{stats[m]['AP_small']['mean']:.5f} ± {stats[m]['AP_small']['std']:.5f}"
        ap_m = f"{stats[m]['AP_medium']['mean']:.5f} ± {stats[m]['AP_medium']['std']:.5f}"
        ap_l = f"{stats[m]['AP_large']['mean']:.5f} ± {stats[m]['AP_large']['std']:.5f}"
        ar100 = f"{stats[m]['AR_max100']['mean']:.5f} ± {stats[m]['AR_max100']['std']:.5f}"
        
        lines.append(f"| {m} | {count} | {ap} | {ap50} | {ap75} | {ap_s} | {ap_m} | {ap_l} | {ar100} |")
        
    summary_path = os.path.join(save_dir, "evaluation_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
        
    print(f"\nSaved markdown evaluation summary to {summary_path}")
    print("\n" + "\n".join(lines) + "\n")
    
    # Save combined results to a JSON file
    summary_json_path = os.path.join(save_dir, "evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved combined JSON evaluation summary to {summary_json_path}")

def main() -> None:
    cfg = PipelineConfig()
    eval_dir = cfg.eval_results_dir
    
    if not os.path.exists(eval_dir):
        print(f"Evaluation directory {eval_dir} does not exist. No results to plot.")
        return
        
    results = load_metrics(eval_dir)
    if not results:
        print(f"No metric JSON files (*_coco_metrics.json) found in {eval_dir}.")
        return
        
    print(f"Loaded {len(results)} metrics files from {eval_dir}.")
    
    stats = group_and_aggregate(results)
    
    # Generate outputs
    plot_ap_comparison(stats, eval_dir)
    plot_individual_seeds(stats, eval_dir)
    generate_markdown_summary(stats, eval_dir)

if __name__ == "__main__":
    main()

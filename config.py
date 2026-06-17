import os
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # W&B Core Setup
    entity: str = "brezemil"
    project: str = "_trainer"
    
    # Model & Data Paths
    model_variant: str = "yolo26s.pt"
    dataset_path: str = r"C:\Users\emilb\_data\_smoketest\dataset.yaml"
    
    # Optional Sweep IDs to plug in
    aug_sweep_id: str = None
    hpo_sweep_id: str = None
    
    # Global Training Constraints
    image_size: int = 128
    max_sweep_runs: int = 2 #75
    sweep_epochs: int = 5 #50
    prod_epochs: int = 5 #250
    device: int = 0
    batch_size: int = 16  # Adjust manually if VRAM is exceeded
    workers: int = 0      # Set to 0 to prevent multiprocessing shared memory issues on Windows
    fraction: float = 1.0 # Fraction of dataset to train on (e.g. 0.01 for 1% data)
    
    # Stock Training Settings (Models & Seeds)
    models: tuple = ("yolo11s.pt", "yolo26s.pt", "rtdetr-l.pt")
    seeds: tuple = (42, 100, 999)
    runs_dir: str = "runs"
    wandb_dir: str = "wandb"
    eval_results_dir: str = "evaluation_results"
    
    # Fixed Loss Parameters (Baseline)
    fixed_loss: dict = None

    def __post_init__(self):
        self.fixed_loss = {"box": 5.63, "cls": 0.56, "dfl": 9.04}
        
        # Resolve logging paths relative to project root (directory of config.py)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self.runs_dir):
            self.runs_dir = os.path.abspath(os.path.join(root_dir, self.runs_dir))
        if not os.path.isabs(self.wandb_dir):
            self.wandb_dir = os.path.abspath(os.path.join(root_dir, self.wandb_dir))
        if not os.path.isabs(self.eval_results_dir):
            self.eval_results_dir = os.path.abspath(os.path.join(root_dir, self.eval_results_dir))
        
    @staticmethod
    def get_best_sweep_config(sweep_id: str, project: str, entity: str) -> dict:
        """Fetches the config of the best performing run from a specified W&B sweep."""
        import wandb
        api = wandb.Api()
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        
        runs = sorted(
            sweep.runs, 
            key=lambda run: run.summary.get("metrics/mAP50-95(B)", 0.0), 
            reverse=True
        )
        if not runs:
            raise ValueError(f"No runs found in sweep {sweep_id}.")
            
        # Strip internal W&B keys starting with '_'
        return {k: v for k, v in runs[0].config.items() if not k.startswith('_')}
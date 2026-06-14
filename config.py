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
    
    # Global Training Constraints
    image_size: int = 256
    max_sweep_runs: int = 10 #75
    sweep_epochs: int = 5 #50
    prod_epochs: int = 250
    device: int = 0
    batch_size: int = 4  # Adjust manually if VRAM is exceeded
    
    
    # Fixed Loss Parameters (Baseline)
    fixed_loss: dict = None

    def __post_init__(self):
        self.fixed_loss = {"box": 5.63, "cls": 0.56, "dfl": 9.04}
        
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
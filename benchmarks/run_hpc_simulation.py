import time
import sys
sys.path.insert(0, "src")
from distributed_trainer.core.trainer import Trainer
from distributed_trainer.data.ingestion import make_synthetic_regression_batch
from omegaconf import OmegaConf

def run_hpc_simulation():
    print("Starting HPC Training Simulation...")
    # Generate dummy configs for 4 nodes
    cfg = OmegaConf.create({
        'training': {
            'num_workers': 4,
            'num_steps': 5,
            'micro_batch_size': 16,
            'learning_rate': 0.001,
            'use_checkpointing': True,
            'use_mixed_precision': True,
            'use_zero': True,
            'seed': 42
        },
        'model': {
            'in_dim': 256,
            'hidden_dim': 1024,
            'out_dim': 10
        }
    })
    
    x, y = make_synthetic_regression_batch(256, 256, 10)
    
    trainer = Trainer(cfg)
    
    start_time = time.time()
    res = trainer.train(x, y)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print(f"Final Loss: {res['loss_history'][-1]:.4f}")

if __name__ == "__main__":
    run_hpc_simulation()

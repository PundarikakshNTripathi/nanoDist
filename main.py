import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from distributed_trainer.core.trainer import Trainer
from distributed_trainer.data.ingestion import make_synthetic_regression_batch

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Initializing nanoDist with the following configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    logger.info("Generating synthetic dataset...")
    x, y = make_synthetic_regression_batch(
        batch_size=cfg.training.batch_size, 
        in_dim=cfg.model.in_dim, 
        out_dim=cfg.model.out_dim, 
        seed=cfg.training.seed
    )
    
    logger.info("Starting distributed training loop...")
    trainer = Trainer(cfg)
    results = trainer.train(x, y)
    
    logger.success(f"Training complete! Final loss: {results['loss_history'][-1]:.6f}")

if __name__ == "__main__":
    main()

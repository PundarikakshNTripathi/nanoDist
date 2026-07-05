import wandb
import os
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf, DictConfig

class TelemetryManager:
    @staticmethod
    def init(cfg: DictConfig):
        load_dotenv()
        if os.getenv("WANDB_API_KEY"):
            wandb.init(project="nanoDist", config=OmegaConf.to_container(cfg, resolve=True))
            logger.info("WandB telemetry initialized.")
        else:
            logger.warning("WANDB_API_KEY not found in .env. Running without telemetry.")
            
    @staticmethod
    def log(metrics: dict):
        if wandb.run is not None:
            wandb.log(metrics)

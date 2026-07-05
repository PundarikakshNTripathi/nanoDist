import hydra
from omegaconf import DictConfig, OmegaConf
from distributed_trainer.core.trainer import Trainer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # trainer = Trainer(cfg)
    # trainer.train()

if __name__ == "__main__":
    main()

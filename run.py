import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    from src.train import train
    from src.utils import utils

    utils.extras(config)
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    return train(config)


if __name__ == "__main__":
    main()

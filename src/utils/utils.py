import logging
import warnings
from typing import Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Log once when running on multiple devices."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for method in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, method, rank_zero_only(getattr(logger, method)))
    return logger


def extras(config: DictConfig) -> None:
    """Apply the explicit warning/debug options from the main configuration."""
    if config.get("ignore_warnings"):
        warnings.filterwarnings("ignore")
    if config.get("debug"):
        with open_dict(config.trainer):
            config.trainer.fast_dev_run = True
    if config.trainer.get("fast_dev_run"):
        config.trainer.accelerator = "cpu"
        config.trainer.devices = 1
        config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("trainer", "model", "datamodule", "train", "callbacks", "logger", "seed"),
    resolve: bool = True,
) -> None:
    tree = rich.tree.Tree("CONFIG", style="dim", guide_style="dim")
    for field in fields:
        section = config.get(field)
        content = OmegaConf.to_yaml(section, resolve=resolve) if isinstance(section, DictConfig) else str(section)
        tree.add(field).add(rich.syntax.Syntax(content, "yaml"))
    rich.print(tree)
    with open("config_tree.txt", "w", encoding="utf-8") as stream:
        rich.print(tree, file=stream)


@rank_zero_only
def log_hyperparameters(config: DictConfig, model: pl.LightningModule, trainer: pl.Trainer) -> None:
    if not trainer.loggers:
        return
    hparams = {
        "config": OmegaConf.to_container(config, resolve=True),
        "model/params_total": sum(p.numel() for p in model.parameters()),
        "model/params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

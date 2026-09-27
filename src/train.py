from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Train TriSA and evaluate the best matching weights-and-graphs checkpoint."""
    pl.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datamodule = hydra.utils.instantiate(config.datamodule)
    model = hydra.utils.instantiate(config.model)
    callbacks = [
        hydra.utils.instantiate(conf)
        for conf in (config.get("callbacks") or {}).values()
        if conf and "_target_" in conf
    ]
    loggers = [
        hydra.utils.instantiate(conf)
        for conf in (config.get("logger") or {}).values()
        if conf and "_target_" in conf
    ]
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )
    utils.log_hyperparameters(config, model, trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    if config.get("test_after_training") and not trainer.fast_dev_run:
        log.info("Testing the best checkpoint!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    if trainer.checkpoint_callback is not None:
        log.info("Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return float(trainer.callback_metrics[optimized_metric])

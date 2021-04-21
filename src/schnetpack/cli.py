import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from typing import List
from schnetpack.utils.script import log_hyperparameters

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

import uuid

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))


@hydra.main(config_path="configs", config_name="train")
def train(config: DictConfig):
    if config.get("print_config"):
        log.info(
            f"Running with the following config:\n {OmegaConf.to_yaml(config, resolve=False)}"
        )

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model, datamodule=datamodule
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)

                # set run_id for AimLogger
                if lg_conf["_target_"] == "aim.pytorch_lightning.AimLogger":
                    from aim import Session

                    sess = Session(
                        repo=l._repo_path,
                        experiment=l._experiment_name,
                        flush_frequency=l._flush_frequency,
                        system_tracking_interval=l._system_tracking_interval,
                        run=config.run_id,
                    )
                    l._aim_session = sess

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.name, config.run_id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(
        config=config, model=model, datamodule=datamodule, trainer=trainer
    )

    # Train the model
    log.info("Starting training.")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test()

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

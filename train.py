from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from model_module import Resnet18Model
from utils import get_model_name


def run_train(cfg, data_module, logger):
    print(OmegaConf.to_yaml(cfg))

    # Инициализация модели
    model = Resnet18Model(cfg, num_classes=10)

    # Получаем уникальное имя модели
    model_name = get_model_name(cfg)
    checkpoint_dir = f"models/{model_name}"

    # Настройка чекпоинтов
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="best",
        save_top_k=1,
        mode="min",
    )

    # Тренер
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    # Логирование гиперпараметров
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    print("Start training model...")
    trainer.fit(model, datamodule=data_module)

    # --- Лучшая модель уже сохранена как best.ckpt ---
    print(
        f"✅ Лучшая модель сохранена как {Path(checkpoint_dir) / 'best.ckpt'}",
    )

    return model, data_module

from pathlib import Path

import pytorch_lightning as pl
import torch

from model_module import Resnet18Model
from utils import get_model_name


def run_infer(cfg, data_module, logger):

    model_name = get_model_name(cfg)
    checkpoint_dir = f"models/{model_name}"
    best_ckpt_path = Path(checkpoint_dir) / "best.ckpt"

    if not best_ckpt_path.is_file():
        raise FileNotFoundError(
            "Не найден чекпоинт best.ckpt. Возможно, модель ещё не обучалась",
        )

    print(f"✅ Загружаем лучшую модель из: {best_ckpt_path}")
    model = Resnet18Model.load_from_checkpoint(best_ckpt_path, cfg=cfg)

    # Тренер
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
    )

    # Инференс
    print("Start testing model...")
    trainer.test(model, datamodule=data_module)

import sys

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import CIFAR10DataModule
from model_module import Resnet18Model


def load_config(overrides=None):
    if overrides is None:
        overrides = []
    # Инициализация Hydra
    with initialize(
        version_base=None, config_path="."
    ):  # Укажите путь к конфигурационным файлам
        # Загрузка конфигурации с переопределениями
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def get_model_name(cfg):
    parts = []
    for key, value in cfg.model.items():
        # Не добавляем сложные объекты, как списки или dict'ы
        if isinstance(value, (list, dict)):
            value = str(value).replace(" ", "")
        parts.append(f"{key}={value}")
    return f"{cfg.baseline_name}_{'_'.join(parts)}"


# Получение аргументов командной строки
cli_args = sys.argv[1:]  # Все аргументы после имени скрипта

# Загрузка конфигурации с учётом CLI-аргументов
cfg = load_config(overrides=cli_args)

# Вывод конфигурации
print(OmegaConf.to_yaml(cfg))

# Логирование в TensorBoard
logger = TensorBoardLogger("tb_logs", name="cifar10_experiment")

# Создание DataModule
data_module = CIFAR10DataModule(
    data_dir=cfg.data_module.data_dir,
    batch_size=cfg.data_module.batch_size,
    valid_size=cfg.data_module.valid_size,
    num_workers=cfg.data_module.num_workers,
    subset_fraction=cfg.data_module.subset_fraction,
)

# Инициализация модели
model = Resnet18Model(cfg, num_classes=10)

# Получаем уникальное имя модели
model_name = get_model_name(cfg)
# print(model_name)

# Создание директории для этой конкретной модели
checkpoint_dir = f"models/{model_name}"

# Настройка чекпоинтов
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename=f"{model_name}-" + "{epoch:02d}-{val_loss:.2f}",
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
    enable_progress_bar=True,  # Включение прогресс-бара
)

# Обучение
print("Start training model...")
trainer.fit(model, datamodule=data_module)

# checkpoint_path = "models/BaselineResnet18-epoch=11-val_loss=0.57.ckpt"
# # Load the model from the checkpoint
# model = ResNet18Model.load_from_checkpoint(checkpoint_path)

# Тестирование
print("Start testing model...")
trainer.test(model, datamodule=data_module)

import sys

from pytorch_lightning.loggers import MLFlowLogger

from data_module import CIFAR10DataModule
from infer import run_infer
from train import run_train
from utils import load_config


def main():
    cli_args = sys.argv[1:]

    # Загрузка конфигурации с переопределениями
    cfg = load_config(overrides=cli_args)

    mode = cfg.mode
    print(f"Режим запуска: {mode}")

    # --- Создание общих объектов ---
    data_module = CIFAR10DataModule(
        data_dir=cfg.data_module.data_dir,
        batch_size=cfg.data_module.batch_size,
        valid_size=cfg.data_module.valid_size,
        num_workers=cfg.data_module.num_workers,
        subset_fraction=cfg.data_module.subset_fraction,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="cifar10_resnet18_experiments",
        tracking_uri=f"http://{cfg.address}",
    )

    if mode in ["train", "train+infer"]:
        run_train(cfg, data_module, mlflow_logger)

    if mode in ["infer", "train+infer"]:
        run_infer(cfg, data_module, mlflow_logger)

    else:
        print("Incorrect mode...")


if __name__ == "__main__":
    main()

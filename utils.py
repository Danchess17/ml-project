from hydra import compose, initialize


def load_config(overrides=None):
    if overrides is None:
        overrides = []
    # Инициализация Hydra
    with initialize(version_base=None, config_path="."):
        # Загрузка конфигурации с переопределениями
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def get_model_name(cfg):
    parts = []
    for key, value in cfg.model.items():
        if isinstance(value, (list, dict)):
            value = str(value).replace(" ", "")
        parts.append(f"{key}={value}")

    # Добавляем subset_fraction в имя модели
    subset_frac = cfg.data_module.subset_fraction
    parts.append(f"subset={subset_frac}")

    return f"{cfg.baseline_name}_{'_'.join(parts)}"

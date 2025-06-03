import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
from torchmetrics import Accuracy

from resnet_model import resnet18


# Функция для загрузки модели из конфига
def load_architecture_from_config(config):
    model_params = config.model

    if config.baseline_name == "Resnet-18":
        model = resnet18(
            num_classes=10,
            oft_layers=model_params.oft_layers,
            r=model_params.r,
            num_matrices=model_params.num_matrices,
            # block_share=model_params.oft.block_share,
        )
    else:
        raise ValueError(f"Неизвестная модель: {config.baseline_name}")

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Resnet18Model(pl.LightningModule):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.save_hyperparameters()  # Сохраняем гиперпараметры
        self.model = load_architecture_from_config(config)
        self.learning_rate = config.learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_results = {"scores": [], "labels": []}
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes
        )  # Метрика для точности

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Сохраняем метки и предсказания
        scores = torch.softmax(output, dim=1).cpu().numpy()
        labels = target.cpu().numpy()

        self.test_results["scores"].extend(scores)
        self.test_results["labels"].extend(labels)

        # Вычисляем точность
        preds = torch.argmax(output, dim=1)
        self.test_accuracy(preds, target)  # Обновляем метрику
        self.log(
            "test_acc",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"scores": scores, "labels": labels}

    def on_fit_start(self):
        # Подсчет и вывод количества обучаемых параметров
        print(self.model)
        num_params = count_parameters(self.model)
        print(f"Точное количество обучаемых параметров: {num_params}")

    def on_test_epoch_end(self):
        all_labels = np.array(self.test_results["labels"])
        all_scores = np.array(self.test_results["scores"])

        predictions = np.argmax(all_scores, axis=1)
        true_labels = np.array(all_labels)

        # Создание новых меток для "своих" и "чужих"
        y_true = true_labels == predictions
        # Используем максимальную вероятность
        y_scores = np.max(all_scores, axis=1)

        # Построение ROC для общей выборки
        fpr_total, tpr_total, _ = roc_curve(y_true, y_scores)
        auc_total = auc(fpr_total, tpr_total)
        print(f"Total AUC: {auc_total:.4f}")

        # Логируем AUC
        self.log("test_auc", auc_total, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

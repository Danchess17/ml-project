import os

import matplotlib.pyplot as plt
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
            odk_layers=model_params.odk_layers,
            r=model_params.r,
            num_matrices=model_params.num_matrices,
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
            task="multiclass",
            num_classes=num_classes,
        )

        # Для накопления средних значений по эпохам
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        acc = self.test_accuracy(torch.argmax(output, dim=1), target)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self):
        # Получаем среднее значение loss за эпоху из логгера
        epoch_mean = self.trainer.callback_metrics.get("train_loss")
        if epoch_mean is not None:
            self.train_losses.append(epoch_mean.item())

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        acc = self.test_accuracy(torch.argmax(output, dim=1), target)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_validation_epoch_end(self):
        # Получаем среднее значение loss за эпоху из логгера
        epoch_mean = self.trainer.callback_metrics.get("val_loss")
        if epoch_mean is not None:
            self.val_losses.append(epoch_mean.item())

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)

        # Сохраняем метки и предсказания
        scores = torch.softmax(output, dim=1).cpu().numpy()
        labels = target.cpu().numpy()

        self.test_results["scores"].extend(scores)
        self.test_results["labels"].extend(labels)

        # Вычисляем точность
        preds = torch.argmax(output, dim=1)
        acc = self.test_accuracy(preds, target)
        self.log(
            "test_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return {"scores": scores, "labels": labels}

    def on_fit_start(self):
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
        y_scores = np.max(all_scores, axis=1)

        # Построение ROC для общей выборки
        fpr_total, tpr_total, _ = roc_curve(y_true, y_scores)
        auc_total = auc(fpr_total, tpr_total)
        print(f"Total AUC: {auc_total:.4f}")

        # Логируем AUC как скаляр
        self.log("test_auc", auc_total, prog_bar=True, logger=True)

        # --- Сохранение графиков ---
        os.makedirs("plots", exist_ok=True)

        self.plot_train_loss()
        self.plot_val_loss()
        self.plot_train_vs_val_loss()

        # --- Логирование артефактов в MLflow ---
        if self.trainer and self.trainer.logger:
            logger = self.trainer.logger
            try:
                files = [
                    "train_loss.png",
                    "val_loss.png",
                    "train_val_loss.png",
                ]
                for file in files:
                    logger.experiment.log_artifact(
                        logger.run_id,
                        local_path=f"plots/{file}",
                        artifact_path="plots",
                    )
                print("✅ Графики успешно залогированы в MLflow")
            except Exception as e:
                print(f"❌ Ошибка при логировании артефактов: {e}")
        else:
            print("⚠️ MLflow logger не найден...")

        # Очищаем данные после тестирования
        self.test_results = {"scores": [], "labels": []}

    def plot_train_loss(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(
            epochs,
            self.train_losses,
            label="Train Loss",
            marker="o",
            color="tab:blue",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Train Loss per Epoch", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/train_loss.png")
        plt.close()

    def plot_val_loss(self):
        epochs = range(1, len(self.val_losses[1:]) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(
            epochs,
            self.val_losses[1:],
            label="Validation Loss",
            marker="s",
            color="tab:orange",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Validation Loss per Epoch", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/val_loss.png")
        plt.close()

    def plot_train_vs_val_loss(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(
            epochs,
            self.train_losses,
            label="Train Loss",
            marker="o",
            color="tab:blue",
        )
        plt.plot(
            epochs,
            self.val_losses[1:],
            label="Validation Loss",
            marker="s",
            color="tab:orange",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Train vs Validation Loss", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/train_val_loss.png")
        plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

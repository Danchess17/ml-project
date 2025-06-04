import tarfile
from pathlib import Path

import dvc.api
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


def download_dataset(
    data_dir: str,
    path_in_repo: str = "data/cifar-10-python.tar.gz",
):
    data_path = Path(data_dir)
    archive_path = data_path / "cifar-10-python.tar.gz"

    if not archive_path.exists():
        print("Downloading dataset via DVC...")
        dvc.api.get_file(
            path=path_in_repo,
            repo=".",  # текущий репозиторий
            out=str(archive_path),  # dvc ожидает строку
        )

    print("Extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_path)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, batch_size, valid_size, num_workers, subset_fraction
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers
        self.subset_fraction = subset_fraction

        # Преобразования для данных
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def prepare_data(self):
        download_dataset(self.data_dir)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Загружаем тренировочные данные
            train_dataset = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform_train
            )
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            subset_size = int(len(indices) * self.subset_fraction)
            indices = indices[:subset_size]

            # Разделяем на тренировочную и валидационную выборки
            split = int(np.floor(self.valid_size * subset_size))
            self.train_indices = indices[split:]
            self.valid_indices = indices[:split]

        if stage == "test" or stage is None:
            # Загружаем тестовые данные
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

    def create_dataloader(self, dataset, indices=None, shuffle=False):
        sampler = SubsetRandomSampler(indices) if indices else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(
                shuffle if sampler is None else False
            ),  # Shuffle только если нет sampler
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.transform_train
        )
        return self.create_dataloader(
            train_dataset,
            indices=self.train_indices,
        )

    def val_dataloader(self):
        val_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.transform_test
        )
        return self.create_dataloader(
            val_dataset,
            indices=self.valid_indices,
        )

    def test_dataloader(self):
        return self.create_dataloader(
            self.test_dataset,
            shuffle=False,
        )

from typing import Optional, Tuple
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.location_handler.base import BaseLocationHandler
from src.data.data_wrapper.base import BaseDataWrapper
from src.data.sampler.base import BaseSampler
from src.data.augmentation.base import BaseAugmentations
import torch


class BaseDataModule(LightningDataModule):
    """Example of LightningDataModule for an image dataset.

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 patch_size: Tuple[int] = None,
                 location_handler: BaseLocationHandler = None,
                 data_wrapper: BaseDataWrapper = None,
                 sampler_train: BaseSampler = None,
                 sampler_val: BaseSampler = None,
                 sampler_test: BaseSampler = None,
                 augmentation_train: BaseAugmentations = None,
                 augmentation_val: BaseAugmentations = None,
                 augmentation_test: BaseAugmentations = None,
                 ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_size = patch_size

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None

        # instantiate sub handler
        self.location_handler: BaseLocationHandler = location_handler
        self.data_wrapper: BaseDataWrapper = data_wrapper

        self.sampler_train: BaseSampler = sampler_train
        self.sampler_val: BaseSampler = sampler_val
        self.sampler_test: BaseSampler = sampler_test
        
        self.augmentation_train: BaseAugmentations = augmentation_train
        self.augmentation_val: BaseAugmentations = augmentation_val
        self.augmentation_test: BaseAugmentations = augmentation_test


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within.
        """
        self.location_handler.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.dataset_train`, `self.dataset_val`, `self.dataset_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`,
        and `trainer.predict()`, so be careful not to execute things like random split twice!
        """
        # set local data directory in data wrapper
        self.data_wrapper.setup_data(self.location_handler.get_local_data_dir())

        # set data for each split sampler
        self.sampler_train.set_dataset(self.data_wrapper.get_train_split())
        self.sampler_val.set_dataset(self.data_wrapper.get_val_split())
        self.sampler_test.set_dataset(self.data_wrapper.get_test_split())

        # create datasets
        self.dataset_train = BaseDataset(sampler=self.sampler_train, augmentation=self.augmentation_train)
        self.dataset_val = BaseDataset(sampler=self.sampler_val, augmentation=self.augmentation_val)
        self.dataset_test = BaseDataset(sampler=self.sampler_test, augmentation=self.augmentation_test)

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(dataset=self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(dataset=self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
    

class BaseDataset(Dataset):
    def __init__(self, sampler: BaseSampler, augmentation: BaseAugmentations):
        self.sampler = sampler
        self.augmentation = augmentation

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample(idx)
        sample = self.augmentation.apply(sample)
        return sample
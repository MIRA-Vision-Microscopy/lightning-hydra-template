from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from src.data.data_wrapper.base import BaseDataSplitWrapper


class BaseSampler(ABC):
    def __init__(self) -> None:
        """Initialize the base sampler."""
        self.sampling_information: Any = None
        self.dataset: Optional[BaseDataSplitWrapper] = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        pass

    def set_dataset(self, dataset: BaseDataSplitWrapper) -> None:
        """Set the dataset for sampling.

        Args:
            dataset (BaseDataSplitWrapper): Dataset to sample from
        """
        self.dataset = dataset

        sampling_information = self.dataset.get_sampling_information()
        self.prepare_sampling(sampling_information)

    @abstractmethod
    def prepare_sampling(self, sampling_information: List) -> None:
        """Prepare sampling by setting up label lookup.

        Args:
            sampling_information (List): Image and label information from dataset
        """
        pass

    @abstractmethod
    def sample(self, idx: int) -> Dict:
        """Sample a patch from the dataset.

        Args:
            idx (int): Index to sample from

        Returns:
            Dict: Sampled patch from the dataset
        """
        pass
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Dict, Optional


class BaseDataSplitWrapper(ABC):
    """Base class for split data wrapper implementations."""
    def __init__(self, data_list: List[Dict], in_memory: bool = False):
        """Initialize the split data wrapper.

        Args:
            data_list (List[Dict]): List of dicts containing data paths and labels
            in_memory (bool, optional): Whether to load data in memory. Defaults to False.
        """
        self.data_list = data_list
        self.in_memory = in_memory

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data_list)

    @abstractmethod
    def get_img_patch(self, idx: int, position: Tuple[int, ...], patch_size: Tuple[int, ...]) -> Dict[str, Any]:
        """Get image patch at the specified index and position.

        Args:
            idx (int): Index of the patch
            position (Tuple[int, ...]): Position coordinates
            patch_size (Tuple[int, ...]): Size of the patch

        Returns:
            Dict[str, Any]: Dictionary containing image patch
        """
        pass

    @abstractmethod
    def get_label_patch(self, idx: int, position: Tuple[int, ...], patch_size: Tuple[int, ...]) -> Dict[str, Any]:
        """Get label patch at the specified index and position.

        Args:
            idx (int): Index of the patch
            position (Tuple[int, ...]): Position coordinates
            patch_size (Tuple[int, ...]): Size of the patch

        Returns:
            Dict[str, Any]: Dictionary containing label information for patch
        """
        pass

    def get_patch(self, idx: int, position: Tuple[int, ...], patch_size: Tuple[int, ...]) -> Dict[str, Any]:
        """Get both image and label patches at the specified index and position.

        Args:
            idx (int): Index of the patch
            position (Tuple[int, ...]): Position coordinates
            patch_size (Tuple[int, ...]): Size of the patch

        Returns:
            Dict[str, Any]: Dictionary containing image and label information for patch
        """
        return {**self.get_img_patch(idx, position, patch_size), **self.get_label_patch(idx, position, patch_size)}

    @abstractmethod
    def get_sampling_information(self) -> List[Dict[str, Any]]:
        """Get information about the labels and images in the dataset, used for sampling policy.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing image and label information
        """
        pass



class BaseDataWrapper(ABC):
    """Base class for data wrapper implementations."""

    def __init__(self, in_memory: bool = False):
        """Initialize the data wrapper.

        Args:
            in_memory (bool, optional): Whether to load data in memory. Defaults to False.
        """
        self.data_dir: Optional[str] = None  # Corrected type hint
        self.in_memory: bool = in_memory

        self.train_split_data: Optional[BaseDataSplitWrapper] = None
        self.val_split_data: Optional[BaseDataSplitWrapper] = None
        self.test_split_data: Optional[BaseDataSplitWrapper] = None
        
    @abstractmethod
    def setup_data(self, data_dir: str) -> None:
        """Setup the data splits, load label information"""
        pass

    def get_train_split(self) -> BaseDataSplitWrapper:
        """Get the training split data.

        Returns:
            BaseDataSplitWrapper: Training split data
        """
        return self.train_split_data
    
    def get_val_split(self) -> BaseDataSplitWrapper:
        """Get the validation split data.

        Returns:
            BaseDataSplitWrapper: Validation split data
        """
        return self.val_split_data
    
    def get_test_split(self) -> BaseDataSplitWrapper:
        """Get the test split data.

        Returns:
            BaseDataSplitWrapper: Test split data
        """
        return self.test_split_data
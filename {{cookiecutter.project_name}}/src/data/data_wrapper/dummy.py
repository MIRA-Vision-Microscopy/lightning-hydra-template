import numpy as np
from typing import List, Dict, Any

from src.data.data_wrapper.base import BaseDataWrapper, BaseDataSplitWrapper


class DummyDataWrapper(BaseDataWrapper):
    """Dummy data wrapper for testing purposes."""

    def __init__(self, in_memory: bool = False):
        """Initialize the dummy data wrapper.

        Args:
            in_memory (bool, optional): Whether to load data in memory. Defaults to False.
        """
        super().__init__(in_memory=in_memory)

    def setup_data(self, data_dir: str) -> None:
        """Setup the dummy data splits, load label information."""
        self.data_dir = data_dir

        # create dummy data
        rng = np.random.default_rng(seed=42)
        data = [{"image": rng.integers(low=0, high=255, size=(512,512,3)).astype(np.uint8),
                 "label": rng.integers(low=0, high=10).astype(np.uint8),
                 "mask": rng.integers(low=0, high=10, size=(512,512), dtype=np.uint8)} for _ in range(100)]
        
        self.train_split_data = DummyDataSplitWrapper(data[:60], in_memory=self.in_memory)
        self.val_split_data = DummyDataSplitWrapper(data[60:80], in_memory=self.in_memory)
        self.test_split_data = DummyDataSplitWrapper(data[80:], in_memory=self.in_memory)


class DummyDataSplitWrapper(BaseDataSplitWrapper):
    """Dummy data split wrapper for testing purposes."""

    def __init__(self, data_list: List[Dict], in_memory: bool = False):
        """Initialize the dummy data split wrapper.

        Args:
            data_list (List[Dict]]): List of dicts containing data paths and labels
            in_memory (bool, optional): Whether to load data in memory. Defaults to False.
        """
        super().__init__(data_list, in_memory=in_memory)
    
    def get_img_patch(self, idx, position, patch_size)-> Dict[str, Any]:
        patch = self.data_list[idx]["image"][position[0]-int(patch_size[0]/2):position[0]+int(patch_size[0]/2), position[1]-int(patch_size[1]/2):position[1]+int(patch_size[1]/2)]
        return {"image": patch}
    
    def get_label_patch(self, idx, position, patch_size)-> Dict[str, Any]:
        mask_patch = self.data_list[idx]["mask"][position[0]-int(patch_size[0]/2):position[0]+int(patch_size[0]/2), position[1]-int(patch_size[1]/2):position[1]+int(patch_size[1]/2)]
        label = self.data_list[idx]["label"]
        return {"label": label, "mask": mask_patch}
    
    def get_sampling_information(self) -> List[Dict]:
        sampling_information = []
        for data in self.data_list:
            sampling_information.append({"img_size": data["image"].shape, "label": data["label"]})

        return sampling_information
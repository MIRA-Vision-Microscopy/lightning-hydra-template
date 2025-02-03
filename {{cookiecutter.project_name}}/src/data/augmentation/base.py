import torch
from abc import ABC
from typing import List
from albumentations import ToFloat
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import BasicTransform


class BaseAugmentations(ABC):
    """Base class for image augmentations."""
    def __init__(self):
        """Initialize the base augmentations.

        Args:
        """

        self.transforms: Compose = self.create_augmentations()

    def create_augmentations(self) -> Compose:
        """Returns a composition of image augmentations.

        Returns:
            Compose: An instance of albumentations Compose with transforms.
        """
        transforms: List[BasicTransform] = [ToFloat(), ToTensorV2()]
        return Compose(transforms)
    
    def apply(self, input_dict: dict) -> dict:
        """Apply augmentations to the input dictionary.

        Args:
            input_dict (dict): Input dictionary containing image and label patches.

        Returns:
            dict: Output dictionary containing augmented image and label patches.
        """

        # collect all transformed entries
        output_dict = {}
        # if labels are contained, conert them to tensor and remove them from the dict
        if "label" in input_dict.keys():
            output_dict["label"] = torch.Tensor([input_dict.pop("label", None)])


        # apply augmentations for all dict entries
        transformed = self.transforms(**input_dict)

        for key in input_dict.keys():
            output_dict[key] = transformed[key]

        return output_dict
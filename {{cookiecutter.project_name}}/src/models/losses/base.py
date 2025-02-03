import torch
from typing import Dict, List, Tuple


class BaseLossContainer(torch.nn.Module):
    """Base class for loss containers."""

    def __init__(self, stage: str) -> None:
        """Initialize the base loss container."""
        super().__init__()

        self.stage = stage
        self.losses: List[Tuple[str, str, str, torch.nn.Module]] = []

    def add_loss(self, name: str, prediction_key: str, batch_key: str, loss: torch.nn.Module) -> None:
        """Add a loss to the container.

        Args:
            name (str): The name of the loss
            prediction_key (str): The key to access the relevant data in the prediction
            batch_key (str): The key to access the relevant data in the batch
            loss (torch.nn.Module): The actual loss instance
        """
        self.losses.append((name, prediction_key, batch_key, loss))
        self.add_module(name, loss)

    def forward(self, prediction, batch) -> Dict[str, torch.Tensor]:
        """Forward pass of the loss container.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss values
        """
        loss_dict = {}
        for name, prediction_key, batch_key, loss in self.losses:
            loss_dict[f"{self.stage}/{name}"] = loss(prediction[prediction_key], batch[batch_key].long())

        # aggregate total loss
        loss_dict[f"{self.stage}/loss"] = sum(loss_dict.values())
        return loss_dict
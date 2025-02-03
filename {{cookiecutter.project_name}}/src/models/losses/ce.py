from torch import nn
from src.models.losses.base import BaseLossContainer


class CELoss(BaseLossContainer):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)

        self.add_loss("ce_loss", "mask", nn.CrossEntropyLoss())
import torchmetrics
from src.models.metrics.base import BaseMetricContainer


class DiceMetric(BaseMetricContainer):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)

        self.add_metric("dice", "probs", "mask", torchmetrics.Dice())
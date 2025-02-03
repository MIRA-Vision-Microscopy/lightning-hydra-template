import torch
from typing import List, Tuple
from torchmetrics import Metric

class BaseMetricContainer(torch.nn.Module):
    """Base container for metrics.

    Metrics are stored as a list of tuples containing:
    - name (str): The name of the metric
    - batch_key (str): The key to access the relevant data in the batch
    - metric (torchmetrics.Metric): The actual metric instance
    """

    def __init__(self, stage: str) -> None:
        super().__init__()
        """Initialize an empty metric container."""
        self.stage = stage
        self.metrics: List[Tuple[str, str, Metric]] = []

    def add_metric(self, name: str, batch_key: str, metric: Metric) -> None:
        """Add a metric to the container.

        Args:
            name (str): The name of the metric
            batch_key (str): The key to access the relevant data in the batch
            metric (torchmetrics.Metric): The actual metric instance
        """
        self.metrics.append((name, batch_key, metric))
        self.add_module(name, metric)

    def update(self, prediction, batch) -> None:
        """Update all metrics with the current batch.

        Args:
            batch: The current batch of data
        """
        for name, batch_key, metric in self.metrics:
            metric.update(prediction, batch[batch_key].long())

    def compute(self) -> dict:
        """Compute all metrics.

        Returns:
            dict: A dictionary containing all metric results with their names as keys
        """
        return {f"{self.stage}/{name}": metric.compute().cpu() for name, _, metric in self.metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        for _, _, metric in self.metrics:
            metric.reset()
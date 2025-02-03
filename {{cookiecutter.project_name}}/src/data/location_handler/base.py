from abc import ABC, abstractmethod
from pathlib import Path


class BaseLocationHandler(ABC):
    """Base class for handling data locations."""

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize the location handler.

        Args:
            data_dir: Path to the data directory (can be string or Path object)
        """
        self.global_data_dir = Path(data_dir)
        self.local_data_dir = None

    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare data directory and files.
        This method should be implemented by child classes.
        This will set the local data directory.
        """
        pass

    def get_local_data_dir(self) -> Path:
        """Get the local data directory.

        Returns:
            Path to the local data directory
        """
        if self.local_data_dir is None:
            raise ValueError("Local data directory has not been set. Call prepare_data() first.")
        return self.local_data_dir
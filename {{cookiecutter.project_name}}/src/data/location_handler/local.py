from src.data.location_handler.base import BaseLocationHandler


class LocalLocationHandler(BaseLocationHandler):
    """Local location handler for handling data locations."""

    def prepare_data(self) -> None:
        """Prepare data directory and files."""
        self.local_data_dir = self.global_data_dir
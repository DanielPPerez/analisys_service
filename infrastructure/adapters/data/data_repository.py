"""Concrete data repository adapter (disk-based placeholder)."""

class DiskDataRepository:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_dataset(self, path):
        # TODO: implement dataset loading from disk
        return None

    def list_samples(self):
        # TODO: implement listing
        return []

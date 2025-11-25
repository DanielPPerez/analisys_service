"""Adapter to persist and load models to/from disk."""

class FileModelRepository:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def save(self, model, path: str):
        # TODO: serialize model
        return True

    def load(self, path: str):
        # TODO: deserialize model
        return None

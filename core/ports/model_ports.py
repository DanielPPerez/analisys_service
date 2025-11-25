"""Ports for model persistence and training orchestration."""

class ModelRepositoryPort:
    """Interface to save/load models and call training routines."""

    def save(self, model, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

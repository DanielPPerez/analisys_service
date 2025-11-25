"""Port interfaces for the core domain."""

from .data_ports import DataLoaderPort
from .model_ports import ModelRepositoryPort
from .analysis_ports import IAnalyzer
from .feedback_ports import IExternalFeedbackGenerator

__all__ = ["ModelRepositoryPort","IAnalyzer", 
    "IExternalFeedbackGenerator", "DataLoaderPort"]

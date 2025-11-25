"""
Puertos de entrada/salida de datos (Interfaces de repositorio).
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import tensorflow as tf

class DataLoaderPort(ABC):
    """Puerto para cargar datasets de entrenamiento/validación."""
    
    @abstractmethod
    def load_datasets(self, source: str, shuffle_buffer: int, prefetch_buffer: Any) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
        """Carga y prepara los datasets."""
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """Retorna el número de clases."""
        pass
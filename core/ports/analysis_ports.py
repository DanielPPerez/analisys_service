# core/ports/analysis_ports.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class IAnalyzer(ABC):
    """
    Puerto para el componente que realiza el análisis geométrico y de trazado de la imagen.
    """
    @abstractmethod
    def analyze_image(self, letter_char: str, image_data: bytes) -> Dict[str, Any]:
        """
        Analiza los datos de la imagen y devuelve un diccionario de métricas.
        El diccionario debe incluir 'score' y 'deviation_code' para cada métrica analizada.
        """
        pass
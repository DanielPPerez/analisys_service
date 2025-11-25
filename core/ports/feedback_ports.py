# core/ports/feedback_ports.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class IExternalFeedbackGenerator(ABC):
    """
    Puerto para el componente que genera feedback personalizado usando servicios externos (e.g., LLMs).
    """
    @abstractmethod
    def generate_personalized_feedback(self, letter_char: str, raw_metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera un feedback constructivo basado en las métricas de análisis.
        Devuelve un diccionario con el feedback de reglas y el feedback de IA.
        """
        pass
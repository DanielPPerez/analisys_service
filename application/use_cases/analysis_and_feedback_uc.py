# application/use_cases/analysis_and_feedback_uc.py
from typing import Dict, Any, Tuple
from core.ports.analysis_ports import IAnalyzer 
from application.use_cases.feedback_generation_uc import FeedbackGenerationUC # Importamos el nuevo Use Case

class AnalysisAndFeedbackUC:
    """
    Caso de uso para analizar una letra y generar el feedback completo.
    """
    def __init__(self, analyzer: IAnalyzer, feedback_uc: FeedbackGenerationUC):
        self.analyzer_adapter: IAnalyzer = analyzer 
        self.feedback_uc: FeedbackGenerationUC = feedback_uc

    def execute(self, letter_char: str, image_data: bytes) -> Tuple[Dict[str, Any], Dict[str, str]]:
        
        # 1. Ejecutar el Análisis
        analysis_metrics = self.analyzer_adapter.analyze_image(
            letter_char=letter_char, 
            image_data=image_data
        )
        
        # 2. Generar el Feedback usando el Use Case de Feedback
        feedback_result = self.feedback_uc.generate(
            letter_char=letter_char,
            raw_metrics=analysis_metrics
        )
        
        return analysis_metrics, feedback_result
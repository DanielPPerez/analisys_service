# application/use_cases/feedback_generation_uc.py
from typing import Dict, Any
from core.ports.feedback_ports import IExternalFeedbackGenerator # Usamos el Puerto
from core.domain_services.rule_based_feedback_generator import RuleBasedFeedbackGenerator # Usamos el generador de reglas como base

class FeedbackGenerationUC:
    """
    Caso de Uso para generar el feedback completo (base de reglas + IA si está disponible).
    """
    def __init__(self, rule_generator: RuleBasedFeedbackGenerator, 
                 external_feedback_generator: IExternalFeedbackGenerator = None):
        
        self.rule_generator = rule_generator
        self.external_feedback_generator = external_feedback_generator # Puede ser None

    def generate(self, letter_char: str, raw_metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera el feedback completo para un carácter analizado.
        """
        # 1. Generar feedback base usando la lógica de reglas (Dominio Puro)
        rule_feedback = self.rule_generator.generate_feedback(raw_metrics)
        
        # 2. Si hay un generador externo (Adaptador LLM), úsalo para enriquecer
        if self.external_feedback_generator:
            # El adaptador LLM ya usa rule_feedback internamente para un mejor prompt
            final_feedback = self.external_feedback_generator.generate_personalized_feedback(
                letter_char=letter_char,
                raw_metrics=raw_metrics # Pasamos las métricas crudas para el prompt
            )
            return final_feedback
        else:
            # Si no hay IA, devolvemos el feedback de reglas
            return {
                "analisis_reglas": rule_feedback,
                "comentario_ia": "Feedback de IA no disponible. Revisa el feedback basado en reglas."
            }
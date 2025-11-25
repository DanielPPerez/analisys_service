# infrastructure/adapters/llm/ai_feedback_adapter.py
import os
from typing import Dict, Any
from openai import OpenAI 
from core.ports.feedback_ports import IExternalFeedbackGenerator 
from core.config import settings
from core.domain_services.rule_based_feedback_generator import RuleBasedFeedbackGenerator


class AIFeedbackAdapter(IExternalFeedbackGenerator): # ¡Implementa el Puerto!
    """
    Adaptador para generar feedback personalizado usando un LLM (OpenAI).
    Implementa el puerto IExternalFeedbackGenerator.
    """
    def __init__(self, api_key: str): # La clave se inyecta desde el exterior
        self.client = OpenAI(api_key=api_key)
        self.rule_generator = RuleBasedFeedbackGenerator()
        
    # Sobrescribe el método abstracto del puerto
    def generate_personalized_feedback(self, letter_char: str, raw_metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera feedback personalizado basándose en métricas y un comentario base de reglas.
        """
        rule_feedback = self.rule_generator.generate_feedback(raw_metrics)
        
        # Extraer métricas detalladas para el prompt
        metrics_summary = []
        metric_names = {
            "score_global": "Puntuación Global",
            "puntuacion_inclinacion": "Inclinación",
            "puntuacion_proporcion": "Proporción",
            "puntuacion_espaciado": "Espaciado Interno",
            "puntuacion_consistencia": "Consistencia del Trazo"
        }
        
        for key, display_name in metric_names.items():
            value = raw_metrics.get(key)
            if value is not None:
                metrics_summary.append(f"- {display_name}: {value}/100")
        
        metrics_text = "\n".join(metrics_summary) if metrics_summary else "No hay métricas disponibles"
        
        prompt = f"""
        Eres un tutor de caligrafía experto y motivador. Tu tarea es crear un comentario personalizado
        y detallado para un estudiante basado en el análisis de su trazo de la letra '{letter_char}'.

        MÉTRICAS DETALLADAS DEL ANÁLISIS:
        {metrics_text}

        ANÁLISIS PRELIMINAR BASADO EN REGLAS:
        - Fortalezas: {rule_feedback['fortalezas']}
        - Áreas de Mejora: {rule_feedback['areas_mejora']}

        INSTRUCCIONES:
        1. Menciona específicamente las métricas más altas como fortalezas
        2. Menciona específicamente las métricas más bajas como áreas de mejora
        3. Si todas las métricas son altas (>=90), felicita al estudiante
        4. Si hay métricas bajas, ofrece consejos específicos y constructivos
        5. Usa un tono motivador y positivo
        6. Sé conciso pero específico (máximo 4 frases)
        7. NO repitas exactamente el análisis de reglas, personaliza el mensaje

        Genera un comentario final personalizado que combine toda esta información.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # O el modelo que prefieras/tengas acceso
                messages=[
                    {"role": "system", "content": "Eres un tutor experto en caligrafía que ofrece feedback constructivo."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            personalized_comment = response.choices[0].message.content.strip()
            
            return {
                "analisis_reglas": rule_feedback,
                "comentario_ia": personalized_comment
            }

        except Exception as e:
            print(f"Error al llamar a la API de OpenAI: {e}")
            return {
                "analisis_reglas": rule_feedback,
                "comentario_ia": f"Lo siento, hubo un error al generar el feedback de IA: {e}. Revisa las fortalezas y mejoras basadas en reglas."
            }
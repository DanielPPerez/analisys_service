# infrastructure/api/v1/endpoints/analysis_endpoint.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import Dict, Any

# Importar el Use Case principal
from application.use_cases.analysis_and_feedback_uc import AnalysisAndFeedbackUC
# Importar las clases CONCRETAS de los adaptadores
from infrastructure.adapters.ml.analysis_adapter import AnalysisAdapter
from application.use_cases.feedback_generation_uc import FeedbackGenerationUC
from infrastructure.adapters.llm.ai_feedback_adapter import AIFeedbackAdapter
from core.domain_services.rule_based_feedback_generator import RuleBasedFeedbackGenerator
import os
from core.config import settings

router = APIRouter()
    
# Función para inyección de dependencias
def get_analysis_uc() -> AnalysisAndFeedbackUC:
    """Inyecta los adaptadores y Use Cases necesarios."""
    
    # 1. Inicializar Adaptadores (Dependencias de bajo nivel)
    analysis_adapter = AnalysisAdapter() 
    
    # Inicializar el generador de feedback LLM solo si hay clave
    feedback_adapter_llm = None
    if settings.API_KEY_OPENAI:
        feedback_adapter_llm = AIFeedbackAdapter(api_key=settings.API_KEY_OPENAI)
    else:
        print("ADVERTENCIA: API_KEY_OPENAI no encontrada. El feedback de IA estará deshabilitado.")

    # 2. Inicializar el Use Case de Feedback (requiere generador de reglas + LLM opcional)
    rule_generator = RuleBasedFeedbackGenerator()
    feedback_uc = FeedbackGenerationUC(
        rule_generator=rule_generator, 
        external_feedback_generator=feedback_adapter_llm
    )
    
    # 3. Inicializar el Use Case Principal
    return AnalysisAndFeedbackUC(
        analyzer=analysis_adapter, 
        feedback_uc=feedback_uc
    )


@router.post("/analyze")
async def analyze_handwriting(
    letter_char: str, 
    file: UploadFile = File(...),
    uc: AnalysisAndFeedbackUC = Depends(get_analysis_uc) # ¡Ahora inyecta el Use Case configurado!
):
    """
    Recibe una imagen y el carácter esperado, devuelve el análisis y el feedback.
    """
    if len(letter_char) != 1:
        raise HTTPException(status_code=400, detail="El parámetro 'letter_char' debe ser un solo carácter.")
        
    # Leer la imagen binaria
    image_data = await file.read()
    
    try:
        # Ejecutar la orquestación a través del Use Case inyectado
        metrics, feedback = uc.execute(letter_char, image_data)
        
        return {
            "caracter_analizado": letter_char,
            "metricas_detalle": metrics,
            "feedback_final": feedback
        }
    except HTTPException as e:
        raise e # Propagar errores de negocio o de configuración (como la API Key)
    except Exception as e:
        # Capturar errores de procesamiento o de IA
        raise HTTPException(status_code=500, detail=f"Error interno al procesar el análisis: {e}")
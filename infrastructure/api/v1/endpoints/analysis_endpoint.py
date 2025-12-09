# infrastructure/api/v1/endpoints/analysis_endpoint.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
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

# Utilidades de validación
from infrastructure.api.v1.endpoints.validators import (
    validate_image_file,
    validate_letter_char,
    sanitize_string
)

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
    letter_char: str = Form(...),
    file: UploadFile = File(...),
    uc: AnalysisAndFeedbackUC = Depends(get_analysis_uc)
):
    """
    Recibe una imagen y el carácter esperado, devuelve el análisis y el feedback.
    
    Validaciones aplicadas:
    - Letra: sanitización, validación de longitud, validación alfanumérica
    - Imagen: extensión, tipo MIME, tamaño (máx 10MB), contenido válido, dimensiones
    - Sanitización de todos los campos de texto
    - Manejo seguro de errores (no expone detalles internos)
    """
    # Validar y sanitizar el carácter de letra
    try:
        validated_letter = validate_letter_char(letter_char)
    except HTTPException:
        # Re-lanzar HTTPException tal cual
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al validar el parámetro 'letter_char'"
        )
    
    print(f"[AnalysisService] Recibida petición de análisis - letra: {validated_letter}, archivo: {file.filename if file.filename else 'N/A'}")
    
    # Leer la imagen binaria
    try:
        image_data = await file.read()
    except Exception as e:
        print(f"[AnalysisService] ERROR al leer el archivo: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al leer el archivo de imagen"
        )
    
    # Validar el archivo de imagen de manera robusta
    # Esta función valida: extensión, tipo MIME, tamaño, contenido válido, dimensiones
    try:
        validated_image, image_metadata = validate_image_file(file, image_data)
        print(f"[AnalysisService] Imagen validada correctamente - tamaño: {image_metadata['size_bytes']} bytes, "
              f"dimensiones: {image_metadata['width']}x{image_metadata['height']}, "
              f"tipo: {image_metadata['content_type']}")
    except HTTPException:
        # Re-lanzar HTTPException tal cual (ya tiene el mensaje apropiado)
        raise
    except Exception as e:
        print(f"[AnalysisService] ERROR inesperado al validar imagen: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al validar el archivo de imagen"
        )
    
    print(f"[AnalysisService] Iniciando análisis para letra: {validated_letter}...")
    
    try:
        # Ejecutar la orquestación a través del Use Case inyectado
        metrics, feedback = uc.execute(validated_letter, image_data)
        
        print(f"[AnalysisService] Análisis completado exitosamente para letra: {validated_letter}")
        
        # Sanitizar campos de texto en la respuesta si es necesario
        # (los use cases deberían hacerlo, pero esto es defensa en profundidad)
        sanitized_metrics = {}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, str):
                    sanitized_metrics[key] = sanitize_string(value, max_length=1000)
                else:
                    sanitized_metrics[key] = value
        else:
            sanitized_metrics = metrics
        
        sanitized_feedback = {}
        if isinstance(feedback, dict):
            for key, value in feedback.items():
                if isinstance(value, str):
                    sanitized_feedback[key] = sanitize_string(value, max_length=2000)
                elif isinstance(value, dict):
                    # Sanitizar valores dentro de diccionarios anidados
                    sanitized_feedback[key] = {
                        k: sanitize_string(v, max_length=2000) if isinstance(v, str) else v
                        for k, v in value.items()
                    }
                else:
                    sanitized_feedback[key] = value
        else:
            sanitized_feedback = feedback
        
        return {
            "caracter_analizado": validated_letter,
            "metricas_detalle": sanitized_metrics,
            "feedback_final": sanitized_feedback
        }
    except HTTPException as e:
        # Propagar errores HTTP tal cual (errores de negocio o configuración)
        print(f"[AnalysisService] ERROR HTTP: {e.status_code} - {e.detail}")
        raise e
    except Exception as e:
        # Capturar errores de procesamiento o de IA
        # No exponer detalles del error interno por seguridad
        error_type = type(e).__name__
        print(f"[AnalysisService] ERROR interno: {error_type}: {str(e)}")
        import traceback
        traceback.print_exc()  # Log detallado para debugging interno
        
        # Mensaje genérico para el cliente
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar el análisis. Por favor, intenta nuevamente."
        )
# core/domain_services/geometric_analysis/analyzer_service.py
import cv2
import numpy as np
import os
from typing import Dict, Any, Optional



def _get_processed_image_for_cv(image_bytes: bytes):
    """Decodifica y prepara la imagen para el análisis CV."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen para análisis CV.")
    return image

def _load_template_image(letter_char: str, templates_dir: str = "data/templates") -> Optional[np.ndarray]:
    """Carga la imagen de la plantilla para un carácter dado."""
    if not os.path.isdir(templates_dir):
        return None
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # Buscar en subdirectorios (lower, upper, numeric)
    for subdir in os.listdir(templates_dir):
        subdir_path = os.path.join(templates_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # Buscar archivo de plantilla que coincida con el carácter
        for filename in os.listdir(subdir_path):
            if not filename.lower().endswith(valid_extensions):
                continue
            
            # Extraer el nombre del carácter del nombre del archivo
            char_name = filename.replace('_template', '').split('.')[0]
            
            if char_name == letter_char:
                template_path = os.path.join(subdir_path, filename)
                try:
                    with open(template_path, 'rb') as f:
                        template_bytes = f.read()
                    template_image = _get_processed_image_for_cv(template_bytes)
                    return template_image
                except Exception:
                    continue
    
    return None

def analyze_errors_cv(image_bytes: bytes, letter_char: Optional[str] = None, 
                      templates_dir: str = "data/templates") -> Dict[str, Any]:
    """
    Ejecuta el análisis detallado de CV. Devuelve valores brutos (e.g., grados de inclinación, ratio WH).
    
    Args:
        image_bytes: Bytes de la imagen del usuario
        letter_char: Carácter esperado (opcional, necesario para análisis que requieren plantilla)
        templates_dir: Directorio donde están las plantillas
    """
    image_cv = _get_processed_image_for_cv(image_bytes)
    
    # Binarizar la imagen para el análisis CV
    _, image_bin = cv2.threshold(image_cv, 127, 255, cv2.THRESH_BINARY)
    
    # --- Lógica de análisis puro del Dominio ---
    # Importaciones movidas aquí para ser cargadas tardíamente
    from core.domain_services.geometric_analysis.inclination_analyzer import analyze_inclination
    from core.domain_services.geometric_analysis.proportion_analyzer import analyze_proportion
    from core.domain_services.geometric_analysis.internal_spacing_analyzer import analyze_internal_spacing
    from core.domain_services.geometric_analysis.stroke_consistency_analyzer import analyze_stroke_consistency
    
    # Análisis que no requieren plantilla
    inclination = analyze_inclination(image_bin)
    consistency = analyze_stroke_consistency(image_bin)
    
    # Análisis que requieren plantilla (si está disponible)
    proportion = {"score": 0.0, "deviation_code": "no_template"}
    spacing = {"score": 0.0, "deviation_code": "no_template"}
    
    if letter_char:
        template_image_bin = _load_template_image(letter_char, templates_dir)
        if template_image_bin is not None:
            _, template_bin = cv2.threshold(template_image_bin, 127, 255, cv2.THRESH_BINARY)
            proportion = analyze_proportion(image_bin, template_bin)
            spacing = analyze_internal_spacing(image_bin, template_bin)
    # -------------------------------------------
    
    return {
        "inclinacion": inclination,         
        "proporcion_wh": proportion,        
        "espaciado": spacing,               
        "consistencia": consistency         
    }

def generate_rules_feedback(score_global: int, cv_results: Dict[str, Any]) -> Dict[str, str]:
    """Genera feedback basado en reglas y la puntuación del modelo."""
    fortalezas = "Buen intento, sigue practicando."
    areas_mejora = "Concéntrate en la forma general de la letra."
    
    if score_global > 85:
        fortalezas = "¡Excelente! La forma es muy similar a la plantilla."
    
    # Simplificación de la lógica de mapeo para el ejemplo
    if cv_results.get('inclinacion', 90) > 10: # Valor de umbral arbitrario
        areas_mejora = "Intenta mantener la letra un poco más vertical."
    elif cv_results.get('proporcion_wh', 0.0) < 0.7:
        areas_mejora = "Asegúrate de que el ancho y alto de la letra sean proporcionales a la plantilla."
        
    return {
        "fortalezas": fortalezas,
        "areas_mejora": areas_mejora
    }
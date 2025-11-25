"""
Analiza y diagnostica la proporción (relación de aspecto) de una letra.
"""
import cv2
import numpy as np
from typing import Dict, Any
# Importación ajustada
from core.domain_services.image_utils import find_main_contour 

def analyze_proportion(user_image_bin: np.ndarray, template_image_bin: np.ndarray) -> Dict[str, Any]:
    """Analiza la proporción comparando la relación de aspecto del bounding box."""
    user_contour = find_main_contour(user_image_bin)
    template_contour = find_main_contour(template_image_bin)

    default_response = {
        "score": 0.0,
        "user_aspect_ratio": 0.0,
        "template_aspect_ratio": 0.0,
        "deviation_code": "no_contour_found"
    }

    if user_contour is None or template_contour is None:
        return default_response

    # Usuario
    _, _, uw, uh = cv2.boundingRect(user_contour)
    user_aspect_ratio = uw / uh if uh > 0 else 0.0

    # Plantilla
    _, _, tw, th = cv2.boundingRect(template_contour)
    template_aspect_ratio = tw / th if th > 0 else 0.0

    if template_aspect_ratio == 0:
        return default_response

    # Error relativo
    error = (user_aspect_ratio - template_aspect_ratio) / template_aspect_ratio
    
    deviation_code = "optima"
    if error > 0.25:
        deviation_code = "demasiado_ancha"
    elif error < -0.25:
        deviation_code = "demasiado_estrecha"

    score = max(0.0, 1.0 - abs(error))
    
    return {
        "score": round(score * 100),
        "user_aspect_ratio": round(user_aspect_ratio, 3),
        "template_aspect_ratio": round(template_aspect_ratio, 3),
        "deviation_code": deviation_code
    }
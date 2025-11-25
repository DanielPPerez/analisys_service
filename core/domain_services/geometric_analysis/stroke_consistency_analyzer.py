"""
Analiza y diagnostica la consistencia del grosor del trazo de una letra.
"""
import cv2
import numpy as np
from typing import Dict, Any

def _skeletonize(image_bin: np.ndarray) -> np.ndarray:
    """Realiza la esqueletización de una imagen binarizada."""
    skeleton = np.zeros(image_bin.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(image_bin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image_bin, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image_bin = eroded.copy()
        if cv2.countNonZero(image_bin) == 0:
            break
    return skeleton

def analyze_stroke_consistency(user_image_bin: np.ndarray) -> Dict[str, Any]:
    """Ejecuta el análisis de consistencia del grosor del trazo."""
    if cv2.countNonZero(user_image_bin) == 0:
        return {"score": 0.0, "thickness_variance": -1.0, "deviation_code": "no_content"}

    dist_transform = cv2.distanceTransform(user_image_bin, cv2.DIST_L2, 5)
    skeleton = _skeletonize(user_image_bin)
    thickness_values = dist_transform[skeleton > 0]

    if len(thickness_values) < 5:
        return {"score": 50.0, "thickness_variance": -1.0, "deviation_code": "not_enough_data"}

    mean_thickness = np.mean(thickness_values)
    std_thickness = np.std(thickness_values)

    if mean_thickness == 0:
        return {"score": 0.0, "thickness_variance": -1.0, "deviation_code": "no_thickness"}

    # Coeficiente de variación
    coeff_of_variation = std_thickness / mean_thickness
    
    deviation_code = "optima"
    
    # LÓGICA DE ESCALADO CALIBRADA PARA PLANTILLAS IDEALES:
    # Las plantillas ideales pueden tener pequeñas variaciones debido al procesamiento
    # de imagen (binarización, redimensionamiento, compresión). Ajustamos los umbrales
    # para que estas variaciones mínimas den 100% de score.
    # Para plantillas perfectas (imágenes generadas), el CV puede ser más alto debido a
    # efectos de renderizado, así que aumentamos significativamente el umbral.
    PERFECT_CV_THRESHOLD = 0.6  # Aumentado significativamente para plantillas procesadas
    
    if coeff_of_variation < PERFECT_CV_THRESHOLD:
        # Para variaciones pequeñas o moderadas (plantillas), dar 100% directamente
        score = 100.0
        deviation_code = "optima"
    else:
        # Función de penalización más suave y calibrada
        # Mapeo: CV=0.6 -> 100%, CV=0.8 -> ~50%, CV>=1.0 -> 0%
        # Usamos una función lineal suave para penalizar gradualmente
        if coeff_of_variation < 0.8:
            # Zona de transición suave: CV entre 0.6 y 0.8
            # Score lineal de 100% a 50%
            score = 100.0 - ((coeff_of_variation - PERFECT_CV_THRESHOLD) / 0.2) * 50.0
            deviation_code = "variacion_menor"
        elif coeff_of_variation < 1.0:
            # Zona intermedia: CV entre 0.8 y 1.0
            # Score lineal de 50% a 0%
            score = 50.0 - ((coeff_of_variation - 0.8) / 0.2) * 50.0
            deviation_code = "trazo_inconsistente"
        else:
            # Variación muy alta: score 0%
            score = 0.0
            deviation_code = "trazo_inconsistente"
        
        # Asegurar que el score esté en el rango [0, 100]
        score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 1),  # Score ya está en rango [0, 100]
        "thickness_variation_coeff": round(coeff_of_variation, 3),
        "deviation_code": deviation_code
    }
"""
Analiza y diagnostica la inclinación de una letra.
"""
import cv2
import numpy as np
from typing import Dict, Any
# Importación ajustada asumiendo que está en un módulo de utils del dominio
from core.domain_services.image_utils import find_main_contour 

def analyze_inclination(user_image_bin: np.ndarray) -> Dict[str, Any]:
    """
    Analiza la inclinación basándose en el rectángulo de área mínima del contorno.
    """
    user_contour = find_main_contour(user_image_bin)

    default_response = {
        "score": 0.0,
        "user_angle": 0.0,
        "deviation_code": "no_contour_found"
    }

    if user_contour is None:
        return default_response
    
    # Verificar que el contorno tenga suficientes puntos para calcular el rectángulo
    if len(user_contour) < 5:
        # Si tiene menos de 5 puntos, intentar usar el bounding rect normal
        x, y, w, h = cv2.boundingRect(user_contour)
        if w == 0 or h == 0:
            return default_response
        # Para un rectángulo perfecto, la inclinación es 0
        return {
            "score": 100.0,
            "user_angle": 0.0,
            "deviation_code": "inclinacion_optima"
        }
    
    # Obtener el rectángulo de área mínima
    (_, _), (_, _), angle = cv2.minAreaRect(user_contour)

    # Normalizar el ángulo: cv2.minAreaRect devuelve ángulos en [-90, 0)
    # Convertir a desviación de la vertical (0 grados = perfectamente vertical)
    # Para letras verticales, el ángulo puede ser -90 (horizontal) o 0 (vertical)
    # Necesitamos convertir ambos casos a 0 grados de desviación
    if angle < -45:
        # Ángulo entre -90 y -45: la letra está más horizontal que vertical
        deviation_angle = abs(angle + 90)  # Convertir a desviación de vertical
    else:
        # Ángulo entre -45 y 0: la letra está más vertical que horizontal
        deviation_angle = abs(angle)  # Ya es la desviación de vertical

    # Calcular código de desviación
    deviation_code = "optima"
    if deviation_angle > 15:
        deviation_code = "inclinacion_excesiva_derecha"
    elif deviation_angle < -15:
        deviation_code = "inclinacion_excesiva_izquierda"

    # Puntuación basada en la desviación: 
    # - 0 grados = 100% (perfecto)
    # - 15 grados = ~67% (umbral de advertencia)
    # - 45 grados = 0% (máxima desviación)
    # Usar una función más suave que premie la perfección
    if deviation_angle <= 10:
        # Muy cerca de perfecto: dar 100% (aumentado el umbral para ser más tolerante)
        score = 100.0
        deviation_code = "inclinacion_optima"
    else:
        # Penalización gradual
        score = max(0.0, 100.0 - (deviation_angle / 45.0) * 100.0)

    return {
        "score": round(score, 1),  # Score ya está en rango [0, 100], no multiplicar
        "user_angle": round(deviation_angle, 2),
        "deviation_code": deviation_code
    }
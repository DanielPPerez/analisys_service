"""
Utilidades puras de Computer Vision (OpenCV) para el análisis de dominio.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

def find_main_contour(image_bin: np.ndarray) -> Optional[np.ndarray]:
    """
    Encuentra el contorno más grande en una imagen binarizada.
    Se asume que la imagen contiene un solo objeto principal (la letra).
    """
    # Si la imagen ya está binarizada, no necesitamos volver a binarizarla
    # Pero asegurémonos de que sea uint8
    if image_bin.dtype != np.uint8:
        image_bin = image_bin.astype(np.uint8)
    
    # Asegurar que la imagen es binaria (solo 0 y 255)
    # Si ya está binarizada, esto no debería cambiar mucho
    _, binary_img = cv2.threshold(image_bin, 127, 255, cv2.THRESH_BINARY)
    
    # RETR_EXTERNAL solo busca los contornos exteriores
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filtrar contornos muy pequeños (ruido)
    min_area = (image_bin.shape[0] * image_bin.shape[1]) * 0.01  # Al menos 1% del área de la imagen
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        return None
        
    # Devuelve el contorno con el área más grande
    return max(valid_contours, key=cv2.contourArea)
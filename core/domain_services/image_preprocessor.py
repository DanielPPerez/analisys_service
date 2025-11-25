# src/ml_core/image_preprocessor.py (Versión Mejorada)
import cv2
import numpy as np

IMG_SIZE = (128, 128)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Toma los bytes de una imagen, la limpia, estandariza y prepara para el modelo.
    """
    try:
        # 1. Decodificar bytes a escala de grises
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None:
            raise ValueError("No se pudo decodificar la imagen. Verifica que sea un formato válido (PNG, JPG, etc.)")
        
        # 2. Binarización (umbral adaptativo e inversión)
        # El trazo será blanco (255) y el fondo negro (0)
        block_size = 31 # Ajustar según el grosor del trazo
        C = 5
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, C
        )

        # 3. Eliminar ruido (opcional, pero recomendado)
        img_denoised = cv2.medianBlur(img_thresh, 3)

        # 4. Centrar la letra en un nuevo canvas
        # Encontrar el contorno más grande (la letra)
        contours, _ = cv2.findContours(img_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No se encontró ningún caracter en la imagen.")
            
        # Asumimos que el contorno más grande es la letra
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Recortar la letra
        char_crop = img_denoised[y:y+h, x:x+w]
        
        # Crear un canvas cuadrado y pegar la letra en el centro
        canvas = np.zeros(IMG_SIZE, dtype=np.uint8)
        
        # Calcular el aspect ratio para redimensionar sin distorsión
        aspect_ratio = w / h
        if aspect_ratio > 1: # Más ancha que alta
            new_w = IMG_SIZE[0]
            new_h = int(new_w / aspect_ratio)
        else: # Más alta que ancha
            new_h = IMG_SIZE[1]
            new_w = int(new_h * aspect_ratio)

        resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calcular posición para pegar
        pad_x = (IMG_SIZE[0] - new_w) // 2
        pad_y = (IMG_SIZE[1] - new_h) // 2
        
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_char
        
        # 5. Normalizar para la red neuronal (valores entre 0 y 1)
        final_img = canvas.astype('float32') / 255.0
        
        # 6. Añadir dimensión de canal (Keras/TF lo requiere)
        final_img = np.expand_dims(final_img, axis=-1)
        
        return final_img

    except Exception as e:
        print(f"Error preprocesando la imagen: {e}")
        raise ValueError("No se pudo procesar la imagen.")

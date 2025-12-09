"""
Utilidades de validación y sanitización para el servicio de análisis.
Implementa validaciones robustas según los estándares de seguridad OWASP.
"""
import os
import io
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status
from PIL import Image

# Constantes de validación de imágenes
ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_WIDTH = 10
MIN_IMAGE_HEIGHT = 10
MAX_IMAGE_WIDTH = 5000
MAX_IMAGE_HEIGHT = 5000


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitiza un string eliminando caracteres peligrosos y aplicando trim.
    
    Args:
        value: String a sanitizar
        max_length: Longitud máxima permitida (opcional)
        
    Returns:
        String sanitizado
    """
    if not value:
        return ""
    
    # Trim de espacios al inicio y final
    value = value.strip()
    
    # Eliminar caracteres de control (ASCII 0-31 y 127)
    value = ''.join(char for char in value if ord(char) >= 32 and ord(char) != 127)
    
    # Limitar longitud si se especifica
    if max_length:
        value = value[:max_length]
    
    return value


def validate_image_file(
    file: UploadFile,
    image_bytes: bytes
) -> Tuple[Image.Image, dict]:
    """
    Valida un archivo de imagen de manera robusta según estándares de seguridad.
    
    Validaciones aplicadas:
    - Presencia del archivo
    - Extensión permitida
    - Tipo MIME permitido
    - Tamaño del archivo (máximo 10MB)
    - Contenido válido de imagen (usando PIL)
    - Dimensiones mínimas y máximas
    
    Args:
        file: Objeto UploadFile de FastAPI
        image_bytes: Bytes del archivo de imagen
        
    Returns:
        Tuple con (objeto Image de PIL, metadata de la imagen)
        
    Raises:
        HTTPException: Si alguna validación falla
    """
    # Validar que el archivo esté presente
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El archivo de imagen es requerido."
        )
    
    # Sanitizar el nombre del archivo
    filename = sanitize_string(file.filename, max_length=255)
    
    # Validar extensión del archivo
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extensión no permitida: {file_extension}. "
                   f"Extensiones válidas: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validar que el archivo no esté vacío
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La imagen recibida está vacía."
        )
    
    # Validar tamaño del archivo
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo es demasiado grande. Tamaño máximo: {MAX_FILE_SIZE / (1024*1024)} MB"
        )
    
    # Validar tipo MIME
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no permitido: {file.content_type}. "
                   f"Tipos permitidos: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # Validar que sea una imagen válida usando PIL
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Verificar que sea una imagen válida
        img.verify()
        # Reabrir después de verify (verify cierra el archivo)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Validar dimensiones mínimas
        if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"La imagen es demasiado pequeña. Dimensiones mínimas: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} píxeles"
            )
        
        # Validar dimensiones máximas
        if img.width > MAX_IMAGE_WIDTH or img.height > MAX_IMAGE_HEIGHT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"La imagen es demasiado grande. Dimensiones máximas: {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT} píxeles"
            )
        
        # Preparar metadata de la imagen
        image_metadata = {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'size_bytes': len(image_bytes),
            'filename': filename,
            'content_type': file.content_type
        }
        
        return img, image_metadata
        
    except HTTPException:
        # Re-lanzar HTTPException tal cual
        raise
    except Exception as e:
        # Cualquier otro error al procesar la imagen
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo no es una imagen válida o está corrupto."
        )


def validate_letter_char(letter_char: str) -> str:
    """
    Valida y sanitiza el carácter de letra.
    
    Args:
        letter_char: String con el carácter a validar
        
    Returns:
        String sanitizado y validado (un solo carácter alfanumérico)
        
    Raises:
        HTTPException: Si el carácter no es válido
    """
    # Sanitizar el input
    letter_sanitized = sanitize_string(letter_char)
    
    # Validar longitud
    if len(letter_sanitized) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El parámetro 'letter_char' debe ser un solo carácter."
        )
    
    # Validar que sea alfanumérico (a-z, A-Z, 0-9)
    if not letter_sanitized.isalnum():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El carácter debe ser alfanumérico (a-z, A-Z, 0-9)."
        )
    
    return letter_sanitized


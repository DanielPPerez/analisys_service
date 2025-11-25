"""
Servicio para cargar plantillas y generar embeddings con el modelo base.
Mejorado: validación de extensiones, logs claros, robustez.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict

from core.domain_services.image_preprocessor import preprocess_image


VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def load_and_embed_templates(
    template_dir: str,
    base_model: tf.keras.Model
) -> Dict[str, np.ndarray]:
    """
    Carga plantillas desde un directorio y genera embeddings.
    Cada archivo debe tener formato: 'a_01.png' o 'b_template.png'
    """
    templates = {}

    if not os.path.isdir(template_dir):
        print(f"⚠️ El directorio de plantillas '{template_dir}' no existe.")
        return {}

    files = [f for f in os.listdir(template_dir) if f.lower().endswith(VALID_EXT)]
    if not files:
        print("⚠️ No se encontraron imágenes de plantilla válidas.")
        return {}

    print(f"Procesando {len(files)} plantillas...")

    for filename in files:
        try:
            char = filename.split("_")[0]  # 'a_template.png' → 'a'
            fpath = os.path.join(template_dir, filename)

            with open(fpath, 'rb') as f:
                image_bytes = f.read()

            # Preprocesar
            img = preprocess_image(image_bytes)

            # Crear embedding
            emb = base_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            templates[char] = emb

        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")

    print(f"Plantillas cargadas correctamente: {len(templates)}")

    return templates

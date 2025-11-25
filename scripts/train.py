"""
Script principal para orquestar el entrenamiento del modelo siamés (Arquitectura Hexagonal).
"""
import os
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

# Intentar importar tqdm, si no está disponible usar una alternativa simple
try:
    from tqdm import tqdm
except ImportError:
    # Fallback simple si tqdm no está instalado (Código omitido por brevedad, se asume que funciona)
    def tqdm(iterable=None, total=None, desc="", unit=""):
        # ... (código de fallback) ...
        if iterable is None:
            class SimpleProgressBar:
                def __init__(self, total, desc, unit):
                    self.total = total
                    self.desc = desc
                    self.unit = unit
                    self.current = 0
                    print(f"\n{desc}: 0/{total} {unit}")
                
                def update(self, n=1):
                    self.current += n
                    if self.total is not None and self.current % max(1, self.total // 20) == 0 or self.current == self.total:
                        print(f"{self.desc}: {self.current}/{self.total} {self.unit}")
                
                def write(self, msg):
                    print(msg)
                
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    if self.total is not None:
                        print(f"{self.desc}: {self.total}/{self.total} {self.unit} completado\n")
                    else:
                        print(f"{self.desc} completado\n")
            return SimpleProgressBar(total, desc, unit)
        return iterable


# =================================================================
# *** PARCHE CRÍTICO PARA RESOLVER ModuleNotFoundError ***
# Añadir el directorio raíz del proyecto a sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =================================================================

# --- Importaciones Hexagonales ---
from infrastructure.adapters.ml.trainer_adapter import TrainingAdapter
from core.data.pair_generator import SiamesePairGenerator
from core.models.siamese_model import build_base_network, build_siamese_model
from core.domain_services.losses import contrastive_loss
from core.domain_services.image_preprocessor import preprocess_image 
from core.metrics.siamese_accuracy import siamese_accuracy_metric
# =================================================================

# ==============================
# CONFIGURACIÓN
# ==============================
DATASET_DIR = "data/variations"   # Directorio principal que contiene 'lower', 'upper', 'numeric'
TEMPLATE_DIR = "data/plantillas"  # Se mantiene por si acaso
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 12
# *** AJUSTE SUGERIDO: Aumentar Epochs y/o Batch Size si los recursos lo permiten ***
EPOCHS = 20 
# BATCH_SIZE = 64 
NUM_CLASSES = 0  

# ==============================
# 1. Cargar imágenes desde carpetas (SIN CAMBIOS, ya estaba bien)
# ==============================
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')

def load_dataset_from_directory(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # ... (código de carga sin cambios) ...
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"ERROR: El directorio del dataset '{dataset_dir}' no existe. Asegúrate de ejecutar los scripts de generación.")

    case_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    all_class_names = []
    
    for case_dir in case_dirs:
        case_path = os.path.join(dataset_dir, case_dir)
        char_dirs = [c for c in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, c))]
        all_class_names.extend(char_dirs)
        
    final_class_names = sorted(list(set(all_class_names)))

    if not final_class_names:
        raise ValueError(f"ERROR: No se encontraron subdirectorios de clases (caracteres) en '{dataset_dir}/[lower|upper|numeric]'.")
    
    print(f"\n=== CARGANDO DATASET ===")
    print(f"Clases combinadas encontradas: {len(final_class_names)}")
    
    images = []
    labels = []
    total_files = 0
    processed_files = 0
    error_count = 0

    for case_dir in case_dirs:
        case_path = os.path.join(dataset_dir, case_dir)
        for class_name in final_class_names:
            class_folder = os.path.join(case_path, class_name)
            if os.path.isdir(class_folder):
                 image_files = [f for f in os.listdir(class_folder) 
                               if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
                 total_files += len(image_files)

    if total_files == 0:
        raise ValueError(f"ERROR: No se encontraron archivos de imagen válidos en las carpetas de caracteres bajo '{dataset_dir}'.")

    print(f"Total de imágenes a procesar: {total_files}\n")

    with tqdm(total=total_files, desc="Cargando imágenes", unit="img") as pbar:
        for idx, class_name in enumerate(final_class_names):
            for case_dir in case_dirs:
                class_folder = os.path.join(dataset_dir, case_dir, class_name)
                
                if os.path.isdir(class_folder):
                    image_files = [f for f in os.listdir(class_folder) 
                                if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]

                    for fname in image_files:
                        fpath = os.path.join(class_folder, fname)
                        try:
                            with open(fpath, 'rb') as f:
                                image_bytes = f.read()

                            img = preprocess_image(image_bytes)
                            
                            if img.ndim == 4 and img.shape[0] == 1:
                                img_squeezed = np.squeeze(img, axis=0)
                            elif img.ndim == 3:
                                img_squeezed = img
                            else:
                                raise ValueError(f"El preprocesamiento devolvió una forma inesperada {img.shape} para {fname}")

                            images.append(img_squeezed)
                            labels.append(idx)
                            processed_files += 1
                            pbar.update(1)

                        except Exception as e:
                            error_count += 1
                            pbar.update(1)
                            if error_count <= 10:
                                pbar.write(f"⚠️  Error al leer {fname}: {e}")
                            continue

    if not images:
         raise ValueError("ERROR: No se pudo cargar ninguna imagen válida del dataset.")

    print(f"\n✅ Dataset cargado exitosamente:")
    print(f"   - Imágenes procesadas: {processed_files}/{total_files}")
    print(f"   - Errores: {error_count}")
    print(f"   - Clases únicas (total): {len(final_class_names)}")
    
    return np.array(images), np.array(labels), final_class_names

# ==============================
# 2. Crear dataset tf.data (SIN CAMBIOS)
# ==============================
def create_tf_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


# ==============================
# 3. Entrenamiento (MODIFICADO)
# ==============================
def train():
    # --- Cargar dataset ---
    try:
        images, labels, class_names = load_dataset_from_directory(DATASET_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return
        
    global NUM_CLASSES
    NUM_CLASSES = len(class_names)

    input_shape_for_model = IMG_SIZE + (1,)
    
    # --- Construir Modelos ---
    base_model = build_base_network(input_shape=input_shape_for_model)
    siamese_model, _ = build_siamese_model(input_shape=input_shape_for_model)

    # --- Crear dataset normal ---
    base_dataset = create_tf_dataset(images, labels)

    # --- Crear dataset de pares ---
    # *** AJUSTE DE MARGEN EN LA PÉRDIDA ***
    LOSS_MARGIN = 0.5  # Margen para la pérdida contrastiva
    
    pair_gen = SiamesePairGenerator(num_classes=NUM_CLASSES)
    train_pairs = pair_gen.create_pairs_dataset(base_dataset)

    # --- Entrenar ---
    print("\n=== PASO 3: ENTRENANDO MODELO ===")
    
    ACCURACY_THRESHOLD = 0.5 
    
    siamese_model.compile(
        loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=LOSS_MARGIN), 
        optimizer="adam",
        metrics=[siamese_accuracy_metric(margin=ACCURACY_THRESHOLD), "accuracy"] 
    )
    
    print(f"Entrenando con Margen de Pérdida={LOSS_MARGIN} y Umbral de Métrica={ACCURACY_THRESHOLD}")
    print("Iniciando el entrenamiento con tf.data.Dataset. Esto puede tardar...")
    siamese_model.fit(train_pairs, epochs=EPOCHS)

    # --- Guardar modelo base y siamés ---
    os.makedirs("data/models", exist_ok=True)
    base_model.save("data/models/base_model.keras")
    siamese_model.save("data/models/siamese_model.keras")

    print("\n=== ENTRENAMIENTO FINALIZADO ===")
    print(f"Modelo base guardado en: data/models/base_model.keras")


if __name__ == "__main__":
    train()
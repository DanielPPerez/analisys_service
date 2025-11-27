"""
Script principal para orquestar el entrenamiento del modelo siamés (Arquitectura Hexagonal).
"""
import os
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from typing import Dict, Any, Tuple, List

# Intentar importar tqdm, si no está disponible usar una alternativa simple
try:
    from tqdm import tqdm
except ImportError:
    # Fallback simple si tqdm no está instalado (Código omitido por brevedad, se asume que funciona)
    def tqdm(iterable=None, total=None, desc="", unit=""):
       
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

# Importaciones para EMNIST
try:
    import tensorflow_datasets as tfds
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("⚠️  tensorflow-datasets no está instalado. EMNIST no estará disponible.")
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

# Configuración para EMNIST
USE_EMNIST = True  # Cambiar a False para usar solo el dataset local
EMNIST_DATASET = "emnist/letters"  # 'emnist/letters' para letras, 'emnist/balanced' para balanceado
EMNIST_SPLIT = "train"  # 'train' o 'test'
EMNIST_SAMPLES_PER_CLASS = 100  # Número de muestras por clase a usar de EMNIST (None para usar todas)  

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
# 1.5. Cargar dataset EMNIST
# ==============================
def load_emnist_dataset(dataset_name: str = "emnist/letters", 
                       split: str = "train",
                       samples_per_class: int = None,
                       target_size: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga el dataset EMNIST y lo preprocesa para el entrenamiento.
    
    Args:
        dataset_name: Nombre del dataset EMNIST ('emnist/letters', 'emnist/balanced', etc.)
        split: División a usar ('train', 'test')
        samples_per_class: Número de muestras por clase a usar (None para usar todas)
        target_size: Tamaño objetivo de las imágenes (alto, ancho)
    
    Returns:
        Tuple de (imágenes, etiquetas, nombres_de_clases)
    """
    if not EMNIST_AVAILABLE:
        raise ImportError("tensorflow-datasets no está disponible. Instálalo con: pip install tensorflow-datasets")
    
    print(f"\n=== CARGANDO DATASET EMNIST: {dataset_name} ===")
    
    # Cargar el dataset
    try:
        ds, ds_info = tfds.load(dataset_name, split=split, with_info=True, as_supervised=True)
    except Exception as e:
        raise RuntimeError(f"Error al cargar EMNIST: {e}. Asegúrate de que el dataset esté disponible.")
    
    # Obtener información del dataset
    num_classes = ds_info.features['label'].num_classes
    class_names = [chr(ord('A') + i) if i < 26 else chr(ord('a') + i - 26) if i < 52 else str(i - 52) 
                   for i in range(num_classes)]
    
    print(f"Clases en EMNIST: {num_classes}")
    print(f"Tamaño del split '{split}': {ds_info.splits[split].num_examples} ejemplos")
    
    images = []
    labels = []
    
    # Contador por clase para limitar muestras
    class_counts = {}
    
    print("Procesando imágenes de EMNIST...")
    with tqdm(total=ds_info.splits[split].num_examples, desc="Cargando EMNIST", unit="img") as pbar:
        for image, label in ds:
            label_int = int(label.numpy())
            
            # Limitar muestras por clase si se especifica
            if samples_per_class is not None:
                if label_int not in class_counts:
                    class_counts[label_int] = 0
                if class_counts[label_int] >= samples_per_class:
                    pbar.update(1)
                    continue
                class_counts[label_int] += 1
            
            # Convertir imagen a numpy y preprocesar
            img = image.numpy()
            
            # EMNIST viene en formato (28, 28) o similar, necesitamos redimensionar
            if len(img.shape) == 2:
                # Imagen en escala de grises
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                # Normalizar a [0, 1] y luego a [0, 255]
                img = (img / 255.0 * 255.0).astype(np.uint8)
            elif len(img.shape) == 3:
                # Imagen con canales
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                if img.shape[2] == 1:
                    img = img.squeeze(2)
            
            # Convertir a bytes y usar el preprocesador del dominio
            try:
                # Convertir a formato de bytes (PNG)
                import io
                from PIL import Image
                img_pil = Image.fromarray(img, mode='L')
                img_bytes = io.BytesIO()
                img_pil.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                # Preprocesar usando el preprocesador del dominio
                img_processed = preprocess_image(img_bytes)
                
                if img_processed.ndim == 4 and img_processed.shape[0] == 1:
                    img_squeezed = np.squeeze(img_processed, axis=0)
                elif img_processed.ndim == 3:
                    img_squeezed = img_processed
                else:
                    pbar.update(1)
                    continue
                
                images.append(img_squeezed)
                labels.append(label_int)
                
            except Exception as e:
                if len(images) < 10:  # Solo mostrar primeros errores
                    pbar.write(f"⚠️  Error al procesar imagen EMNIST: {e}")
                pbar.update(1)
                continue
            
            pbar.update(1)
            
            # Detener si hemos alcanzado el límite de muestras por clase
            if samples_per_class is not None and len(images) >= num_classes * samples_per_class:
                break
    
    if not images:
        raise ValueError("No se pudo cargar ninguna imagen válida de EMNIST.")
    
    print(f"\n✅ EMNIST cargado exitosamente:")
    print(f"   - Imágenes procesadas: {len(images)}")
    print(f"   - Clases: {num_classes}")
    
    return np.array(images), np.array(labels), class_names

# ==============================
# 1.6. Combinar datasets
# ==============================
def combine_datasets(local_images: np.ndarray, local_labels: np.ndarray, local_classes: List[str],
                     emnist_images: np.ndarray, emnist_labels: np.ndarray, emnist_classes: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Combina el dataset local con EMNIST, mapeando las clases correctamente.
    """
    print("\n=== COMBINANDO DATASETS ===")
    
    # Crear mapeo de clases EMNIST a índices locales
    # EMNIST tiene letras en orden A-Z, a-z, 0-9
    # Necesitamos mapear a nuestros nombres de clases
    combined_classes = sorted(list(set(local_classes + emnist_classes)))
    class_to_idx = {cls: idx for idx, cls in enumerate(combined_classes)}
    
    # Remapear etiquetas del dataset local
    local_labels_remapped = np.array([class_to_idx[local_classes[label]] for label in local_labels])
    
    # Remapear etiquetas de EMNIST
    emnist_labels_remapped = np.array([class_to_idx[emnist_classes[label]] for label in emnist_labels])
    
    # Combinar imágenes y etiquetas
    combined_images = np.concatenate([local_images, emnist_images], axis=0)
    combined_labels = np.concatenate([local_labels_remapped, emnist_labels_remapped], axis=0)
    
    print(f"✅ Datasets combinados:")
    print(f"   - Total de imágenes: {len(combined_images)}")
    print(f"   - Total de clases: {len(combined_classes)}")
    print(f"   - Imágenes locales: {len(local_images)}")
    print(f"   - Imágenes EMNIST: {len(emnist_images)}")
    
    return combined_images, combined_labels, combined_classes

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
    # --- Cargar dataset local ---
    local_images, local_labels, local_class_names = None, None, None
    try:
        local_images, local_labels, local_class_names = load_dataset_from_directory(DATASET_DIR)
        print(f"✅ Dataset local cargado: {len(local_images)} imágenes, {len(local_class_names)} clases")
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️  Error al cargar dataset local: {e}")
        if not USE_EMNIST:
            print("❌ No se puede continuar sin dataset local y EMNIST está deshabilitado.")
            return
    
    # --- Cargar dataset EMNIST si está habilitado ---
    emnist_images, emnist_labels, emnist_class_names = None, None, None
    if USE_EMNIST and EMNIST_AVAILABLE:
        try:
            emnist_images, emnist_labels, emnist_class_names = load_emnist_dataset(
                dataset_name=EMNIST_DATASET,
                split=EMNIST_SPLIT,
                samples_per_class=EMNIST_SAMPLES_PER_CLASS,
                target_size=IMG_SIZE
            )
            print(f"✅ Dataset EMNIST cargado: {len(emnist_images)} imágenes, {len(emnist_class_names)} clases")
        except Exception as e:
            print(f"⚠️  Error al cargar EMNIST: {e}")
            if local_images is None:
                print("❌ No se puede continuar sin ningún dataset.")
                return
    
    # --- Combinar datasets si ambos están disponibles ---
    if local_images is not None and emnist_images is not None:
        images, labels, class_names = combine_datasets(
            local_images, local_labels, local_class_names,
            emnist_images, emnist_labels, emnist_class_names
        )
    elif local_images is not None:
        images, labels, class_names = local_images, local_labels, local_class_names
    elif emnist_images is not None:
        images, labels, class_names = emnist_images, emnist_labels, emnist_class_names
    else:
        print("❌ No hay datasets disponibles para entrenar.")
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
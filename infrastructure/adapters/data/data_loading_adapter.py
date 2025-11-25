"""
Adaptador de infraestructura para la carga de datos usando tensorflow_datasets (TFDS).
Implementa el puerto `DataLoaderPort` (que aún debes definir en core/ports/data_ports.py).
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, Dict, Any, Optional

# Necesitas definir los puertos en core/ports/data_ports.py
from core.ports.data_ports import DataLoaderPort 


class DataLoadingAdapter(DataLoaderPort):
    """Adaptador que implementa la carga de datos EMNIST usando TFDS."""
    
    def __init__(self, img_size: Tuple[int, int], batch_size: int):
        self.img_size = img_size
        self.batch_size = batch_size
        self._num_classes = None
        
    def _preprocess_fn(self, record: Dict[str, Any], is_training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        """Función de preprocesamiento para los datos de TFDS."""
        image = record['image']
        label = record['label']
        
        # 1. Binarización (Si la imagen de entrada no es binaria)
        # Como EMNIST tiene valores, los binarizamos a 0/255 (letra blanca/fondo negro)
        # Esto es crucial para que los analizadores CV funcionen correctamente.
        image = tf.cast(image, tf.float32)
        binary_image = tf.where(image > 0.5, 255.0, 0.0) # Binarizar
        
        # 2. Redimensionar
        image = tf.image.resize(binary_image, self.img_size)
        
        # 3. Normalizar [0, 1] y añadir canal (128x128x1)
        image = image / 255.0
        image = tf.expand_dims(image, axis=-1)
        
        # 4. Re-escalar a [0, 255] y convertir a uint8 si la capa CV lo necesita más tarde, 
        # o dejarlo como float32 [0, 1] para el modelo siamés. 
        # Para el modelo siamés (que espera float32 [0,1] en su red base, pero que ya
        # maneja la normalización), solo necesitamos el preprocesamiento a (128,128,1)
        # y asegurar que el valor sea 1 para la letra y 0 para el fondo.
        
        # Convertir a uint8 para CV si fuera necesario, pero para el modelo float32 [0,1]
        # ya está casi listo, solo necesitamos el formato correcto.
        
        return image, label

    def load_datasets(self, source: str, shuffle_buffer: int, prefetch_buffer: tf.data.AUTOTUNE) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
        """Carga y prepara los datasets de entrenamiento y prueba."""
        
        # Cargar el dataset (ej. 'emnist/balanced')
        (ds_train, ds_test), ds_info = tfds.load(
            source,
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        
        self._num_classes = ds_info.features['label'].num_classes

        # Mapear y configurar el pipeline
        ds_train = ds_train.map(
            lambda img, label: self._preprocess_fn({'image': img, 'label': label}, is_training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_train = ds_train.shuffle(shuffle_buffer).batch(self.batch_size).prefetch(prefetch_buffer)
        
        ds_test = ds_test.map(
            lambda img, label: self._preprocess_fn({'image': img, 'label': label}, is_training=False),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_test = ds_test.batch(self.batch_size).prefetch(prefetch_buffer)
        
        return ds_train, ds_test, self._num_classes

    def get_num_classes(self) -> int:
        """Retorna el número de clases conocido."""
        if self._num_classes is None:
            raise ValueError("Datasets no cargados. Llama a load_datasets() primero.")
        return self._num_classes
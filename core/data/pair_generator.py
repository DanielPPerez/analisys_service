"""
Generador de pares para entrenamiento de redes siamesas.
"""
import tensorflow as tf
from typing import Tuple


class SiamesePairGenerator:
    """Genera pares de imágenes para entrenamiento de redes siamesas."""
    
    def __init__(self, num_classes: int, buffer_size: int = 10000):
        """Inicializa el generador de pares."""
        self.num_classes = num_classes
        self.buffer_size = buffer_size
    
    def create_pairs_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Crea un dataset de pares a partir de un dataset de imágenes."""
        def create_pairs_batch(images, labels):
            """Crea pares (positivo y negativo) para un batch."""
            batch_size = tf.shape(images)[0]
            batch_size_int32 = tf.cast(batch_size, tf.int32)
            labels = tf.cast(labels, tf.int32)
            indices = tf.range(batch_size_int32, dtype=tf.int32)
            
            # Lógica para encontrar pares positivos y negativos
            labels_i = tf.expand_dims(labels, 1)
            labels_j = tf.expand_dims(labels, 0)
            same_class_matrix = tf.equal(labels_i, labels_j)
            
            # --- Positivos ---
            eye = tf.eye(batch_size_int32, dtype=tf.bool)
            same_no_diag = tf.logical_and(same_class_matrix, tf.logical_not(eye))
            same_float = tf.cast(same_no_diag, tf.float32) * 2.0 - 1.0
            max_vals = tf.reduce_max(same_float, axis=1)
            argmax_indices = tf.argmax(same_float, axis=1, output_type=tf.int32)
            has_match = tf.greater(max_vals, 0.0)
            positive_indices = tf.where(has_match, argmax_indices, indices)
            positive_indices = tf.cast(positive_indices, tf.int32)
            
            # --- Negativos ---
            different_matrix = tf.logical_not(same_class_matrix)
            different_float = tf.cast(different_matrix, tf.float32)
            max_vals_neg = tf.reduce_max(different_float, axis=1)
            argmax_indices_neg = tf.argmax(different_float, axis=1, output_type=tf.int32)
            has_diff = tf.greater(max_vals_neg, 0.0)
            random_fallback = tf.random.uniform([batch_size_int32], 0, batch_size_int32, dtype=tf.int32)
            negative_indices = tf.where(has_diff, argmax_indices_neg, random_fallback)
            negative_indices = tf.cast(negative_indices, tf.int32)
            
            # Obtener imágenes y concatenar
            positive_img2 = tf.gather(images, positive_indices)
            negative_img2 = tf.gather(images, negative_indices)
            
            image1_batch = tf.concat([images, images], axis=0)
            image2_batch = tf.concat([positive_img2, negative_img2], axis=0)
            pair_labels = tf.concat([
                tf.ones(batch_size, dtype=tf.float32), # 1.0 para positivo
                tf.zeros(batch_size, dtype=tf.float32) # 0.0 para negativo
            ], axis=0)
            
            return (image1_batch, image2_batch), pair_labels
        
        # Mapear y mezclar
        paired_dataset = dataset.map(
            create_pairs_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        paired_dataset = paired_dataset.shuffle(buffer_size=self.buffer_size)
        
        return paired_dataset
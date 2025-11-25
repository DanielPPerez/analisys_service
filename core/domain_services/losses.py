"""
Funciones de pérdida para entrenamiento de redes siamesas.
"""
import tensorflow as tf
from tensorflow.keras import backend as K


def contrastive_loss(y_true, y_pred, margin: float = 1.0):
    """Calcula la pérdida contrastiva."""
    y_true = tf.cast(y_true, tf.float32)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
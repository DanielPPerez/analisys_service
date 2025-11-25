"""
Definición de la arquitectura de la red siamesa.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
from typing import Tuple


def build_base_network(input_shape: Tuple[int, int, int] = (128, 128, 1)):
    """Construye la red CNN base para generar embeddings."""
    input_layer = Input(shape=input_shape, name="base_input")
    
    # Capas convolucionales
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    
    # Capas densas
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Capa de embedding final
    embedding = Dense(256, activation=None, name="embedding")(x)
    
    return Model(input_layer, embedding, name="base_network")


def euclidean_distance(vectors):
    """Calcula la distancia euclidiana entre dos vectores de embedding."""
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def build_siamese_model(input_shape: Tuple[int, int, int] = (128, 128, 1)):
    """Construye el modelo siamés completo para entrenamiento."""
    base_network = build_base_network(input_shape)
    
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, name="distance")([processed_a, processed_b])
    
    siamese_model = Model([input_a, input_b], distance, name="siamese_model")
    
    return siamese_model, base_network
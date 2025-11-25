"""
Métrica personalizada para redes siamesas.
"""
import tensorflow as tf
import tensorflow.keras.backend as K

def siamese_accuracy_metric(margin=1.0):
    """
    Métrica de precisión basada en si la distancia predicha está por debajo del umbral
    para pares positivos y por encima para pares negativos.
    """
    # *** AJUSTE DE UMBRAL ***
    # Cambiado de margin/2.0 (0.5) a un valor más alto (0.7) para el inicio del entrenamiento.
    threshold = margin 
    
    def acc_fn(y_true, y_pred):
        # y_true: 1 para positivo, 0 para negativo
        # y_pred: Distancia euclidiana predicha
        
        # Usaremos el umbral definido arriba
        acc_threshold = tf.cast(threshold, tf.float32)
        
        # Predicción: Si la distancia es menor al umbral, es un 'match' (predicción = 1)
        predicted_match = tf.cast(y_pred < acc_threshold, tf.float32)
        
        # Comparar la predicción con la verdad (y_true)
        # Para Positivo (y_true=1): acierta si predicted_match=1 (distancia < threshold)
        # Para Negativo (y_true=0): acierta si predicted_match=0 (distancia >= threshold)
        correct_predictions = tf.where(
            tf.equal(y_true, 1.0),
            tf.cast(tf.equal(predicted_match, 1.0), tf.float32), # Correcto si predijo match
            tf.cast(tf.equal(predicted_match, 0.0), tf.float32)  # Correcto si predijo NO match
        )
        
        return tf.reduce_mean(correct_predictions)

    return acc_fn
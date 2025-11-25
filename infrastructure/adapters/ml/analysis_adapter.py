# infrastructure/adapters/ml/analysis_adapter.py
import numpy as np
import tensorflow as tf
import os
from typing import Dict, Any

# Importaciones de Puertos y Dominio (dependencia 'hacia adentro')
from core.ports.analysis_ports import IAnalyzer # Hereda del puerto
from core.domain_services.image_preprocessor import preprocess_image
# Ajuste: Usaremos generate_rules_feedback directamente, ya que el puntaje final se calculará aquí
# y el feedback de reglas es lógica de Dominio.
from core.domain_services.geometric_analysis.analyzer_service import analyze_errors_cv 
from core.domain_services.rule_based_feedback_generator import RuleBasedFeedbackGenerator # Para usar la lógica de mapeo de reglas
from core.models.siamese_model import build_base_network, euclidean_distance


class AnalysisAdapter(IAnalyzer): # ¡Implementa el Puerto!
    """
    Adaptador que conecta la capa de aplicación con el modelo de Machine Learning
    y el análisis geométrico del Dominio. Implementa IAnalyzer.
    """
    def __init__(self, base_model_path: str = "data/models/base_model.keras", 
                 siamese_model_path: str = "data/models/siamese_model.keras",
                 templates_dir: str = "data/templates"):
        
        self.base_model_path = base_model_path
        self.siamese_model_path = siamese_model_path
        self.templates_dir = templates_dir
        self.base_model = self._load_model()
        self.templates = self._load_templates()
        self.rule_generator = RuleBasedFeedbackGenerator() # Para generar el feedback de reglas

    def _load_model(self) -> tf.keras.Model:
        """Carga el modelo base entrenado. Intenta cargar base_model.keras primero."""
        # Intentar cargar el modelo base directamente (preferido)
        if os.path.exists(self.base_model_path):
            try:
                model = tf.keras.models.load_model(self.base_model_path)
                print(f"✅ Modelo base cargado desde {self.base_model_path}")
                return model
            except Exception as e:
                print(f"⚠️  Error al cargar modelo base: {e}. Intentando alternativa...")
        
        # Si no existe base_model.keras, intentar extraer la red base del modelo siamés
        if os.path.exists(self.siamese_model_path):
            try:
                # Cargar el modelo siamés completo con la función euclidean_distance registrada
                siamese_model = tf.keras.models.load_model(
                    self.siamese_model_path,
                    custom_objects={
                        'euclidean_distance': euclidean_distance
                    }
                )
                # Extraer la red base del modelo siamés
                # La red base es la primera capa interna del modelo siamés
                base_layer = siamese_model.get_layer('base_network')
                if base_layer is None:
                    # Si no tiene nombre 'base_network', buscar la primera capa que sea un modelo
                    for layer in siamese_model.layers:
                        if isinstance(layer, tf.keras.Model):
                            base_layer = layer
                            break
                
                if base_layer is None:
                    raise ValueError("No se pudo encontrar la red base en el modelo siamés")
                
                # Construir un nuevo modelo solo con la red base
                base_input = tf.keras.Input(shape=(128, 128, 1), name="base_input")
                base_output = base_layer(base_input)
                base_model = tf.keras.Model(base_input, base_output, name="base_network")
                
                print(f"✅ Red base extraída del modelo siamés desde {self.siamese_model_path}")
                return base_model
            except Exception as e:
                print(f"⚠️  Error al cargar modelo siamés: {e}. Intentando construir red base desde cero...")
        
        # Último recurso: construir la red base desde cero (sin pesos entrenados)
        print("⚠️  ADVERTENCIA: No se encontraron modelos guardados. Construyendo red base sin pesos entrenados.")
        base_model = build_base_network()
        return base_model


    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Carga y procesa embeddings de plantillas usando el modelo base."""
        templates = {}
        if not os.path.isdir(self.templates_dir):
            print(f"ADVERTENCIA: El directorio de plantillas '{self.templates_dir}' no existe.")
            return {}
        
        # Buscar archivos de plantilla en subdirectorios (lower, upper, numeric)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        for subdir in os.listdir(self.templates_dir):
            subdir_path = os.path.join(self.templates_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Buscar archivos de plantilla en este subdirectorio
            for filename in os.listdir(subdir_path):
                if not filename.lower().endswith(valid_extensions):
                    continue
                
                # Extraer el nombre del carácter del nombre del archivo
                # Ejemplo: "a_template.png" -> "a", "A_template.png" -> "A"
                char_name = filename.replace('_template', '').split('.')[0]
                
                # Evitar duplicados (si ya existe, usar el primero encontrado)
                if char_name in templates:
                    continue
                
                # Cargar y procesar la imagen de la plantilla
                template_path = os.path.join(subdir_path, filename)
                try:
                    with open(template_path, 'rb') as f:
                        image_bytes = f.read()
                    
                    # Preprocesar la imagen
                    img_processed = preprocess_image(image_bytes)
                    
                    # Asegurar la forma esperada (1, H, W, 1)
                    img_expanded = np.expand_dims(img_processed, axis=0)
                    img_expanded = np.expand_dims(img_expanded, axis=-1)
                    
                    # Generar embedding usando el modelo base
                    embedding = self.base_model.predict(img_expanded, verbose=0)[0]
                    templates[char_name] = embedding
                    
                except Exception as e:
                    print(f"⚠️  Error al procesar plantilla {filename}: {e}")
                    continue
        
        print(f"✅ Se cargaron {len(templates)} embeddings de plantillas desde {self.templates_dir}")
        return templates

    def _distance_to_score(self, distance: float, max_distance=15.0) -> int:
        """Convierte la distancia euclidiana a puntuación (Lógica de Dominio)."""
        similarity = max(0, 1 - (distance / max_distance))
        return int(min(100, similarity * 100)) # Puntuación máxima 100
    
    # MÉTODO PRINCIPAL QUE CUMPLE EL CONTRATO IAnalyzer
    def analyze_image(self, letter_char: str, image_data: bytes) -> Dict[str, Any]:
        """
        Analiza los datos de la imagen para obtener puntajes y códigos de desviación.
        """
        # 1. Obtener el embedding pre-calculado de la plantilla
        template_embedding = self.templates.get(letter_char)
        if template_embedding is None:
            # Este error debe ser capturado en el Use Case o superior
            return {"error": "template_missing", "score": 0, "deviation_code": "template_missing"}

        # 2. Preprocesar y extraer el embedding del usuario (Uso de Dominio)
        try:
            user_img_processed = preprocess_image(image_data)
            # Asegurar la forma esperada (1, H, W, 1)
            user_img_expanded = np.expand_dims(user_img_processed, axis=0) 
            user_img_expanded = np.expand_dims(user_img_expanded, axis=-1) 
            
            user_embedding = self.base_model.predict(user_img_expanded)[0]
        except ValueError as e:
            # Error de preprocesamiento (imagen vacía, no decodificable, etc.)
            return {"error": str(e), "score": 0, "deviation_code": "no_contour_found"}

        # 3. Calcular distancia y puntaje global
        distance = np.linalg.norm(user_embedding - template_embedding)
        score_global = self._distance_to_score(float(distance))
        
        # 3.5. Detectar si la imagen es idéntica a la plantilla (distancia muy pequeña)
        # Si la distancia es muy pequeña, es probable que sea la misma imagen
        IDENTICAL_THRESHOLD = 0.1  # Umbral muy bajo para detectar imágenes idénticas
        is_identical = distance < IDENTICAL_THRESHOLD
        
        # 4. Análisis detallado CV (Llamada al servicio de Dominio)
        try:
            detalles_cv = analyze_errors_cv(image_data, letter_char=letter_char, templates_dir=self.templates_dir)
            
            # Si la imagen es idéntica a la plantilla, forzar todas las métricas a 100
            if is_identical:
                detalles_cv = {
                    "inclinacion": {"score": 100.0, "deviation_code": "inclinacion_optima"},
                    "proporcion_wh": {"score": 100.0, "deviation_code": "proporcion_optima"},
                    "espaciado": {"score": 100.0, "deviation_code": "espaciado_interno_optimo"},
                    "consistencia": {"score": 100.0, "deviation_code": "consistencia_optima"}
                }
                score_global = 100  # También forzar el score global a 100
        except Exception as e:
             print(f"⚠️  Error en análisis CV: {e}")
             detalles_cv = {
                "inclinacion": {"score": 0.0, "deviation_code": "error"}, 
                "proporcion_wh": {"score": 0.0, "deviation_code": "error"}, 
                "espaciado": {"score": 0.0, "deviation_code": "error"}, 
                "consistencia": {"score": 0.0, "deviation_code": "error"}
             }

        # 5. Extraer valores de los diccionarios de análisis CV
        inclinacion_data = detalles_cv.get('inclinacion', {})
        proporcion_data = detalles_cv.get('proporcion_wh', {})
        espaciado_data = detalles_cv.get('espaciado', {})
        consistencia_data = detalles_cv.get('consistencia', {})
        
        # 6. Generar feedback basado en reglas (Lógica de Dominio)
        feedback_base = self.rule_generator.generate_feedback({
            "inclinacion": {
                "score": inclinacion_data.get('score', 0.0),
                "deviation_code": inclinacion_data.get('deviation_code', 'no_data')
            },
            "proporcion": {
                "score": proporcion_data.get('score', 0.0),
                "deviation_code": proporcion_data.get('deviation_code', 'no_data')
            },
            "espaciado": {
                "score": espaciado_data.get('score', 0.0),
                "deviation_code": espaciado_data.get('deviation_code', 'no_data')
            },
            "consistencia": {
                "score": consistencia_data.get('score', 0.0),
                "deviation_code": consistencia_data.get('deviation_code', 'no_data')
            }
        })
        
        # 7. Formatear la salida final (la puntuación y los códigos son lo importante para el LLM)
        return {
            "score_global": score_global,
            "puntuacion_inclinacion": inclinacion_data.get('score', 0.0),
            "puntuacion_proporcion": proporcion_data.get('score', 0.0),
            "puntuacion_espaciado": espaciado_data.get('score', 0.0),
            "puntuacion_consistencia": consistencia_data.get('score', 0.0),
            "deviation_code_global": feedback_base.get("areas_mejora", "").split('.')[0] if feedback_base.get("areas_mejora") != "¡Tu letra es prácticamente perfecta! No tenemos ninguna sugerencia por ahora." else "proporcion_optima",
            "fortalezas_base": feedback_base.get("fortalezas", ""),
            "areas_mejora_base": feedback_base.get("areas_mejora", "")
        }
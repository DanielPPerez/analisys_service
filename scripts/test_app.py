import streamlit as st
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Any

# --- AGREGAR EL DIRECTORIO RAÍZ AL PATH ---
# Esto permite que Python encuentre el módulo 'core' cuando se ejecuta desde scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTAR TUS MÓDULOS DE ANÁLISIS ---
# AJUSTA ESTAS IMPORTACIONES según la estructura real de tus archivos
from core.domain_services.geometric_analysis.inclination_analyzer import analyze_inclination
from core.domain_services.geometric_analysis.proportion_analyzer import analyze_proportion
from core.domain_services.geometric_analysis.internal_spacing_analyzer import analyze_internal_spacing
from core.domain_services.geometric_analysis.stroke_consistency_analyzer import analyze_stroke_consistency
# Importa tu generador de feedback. Asumiré que se llama RuleBasedFeedbackGenerator
# y que tiene un método 'generate_feedback' que toma el diccionario de resultados CV.
from core.domain_services.rule_based_feedback_generator import RuleBasedFeedbackGenerator 

# --- CONFIGURACIÓN DE RUTAS ---
MODEL_PATH = "data/models/base_model.keras"
TEMPLATE_BASE_DIR = "data/templates"
IMG_SIZE = (128, 128)
SIMILARITY_THRESHOLD = 2.0 # Distancia que se mapea a un score de 0. (Ajustar si es necesario)
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')

# --- 1. Funciones Auxiliares de Preprocesamiento (Siamés y CV) ---

def preprocess_for_siam_model(img_path: str) -> np.ndarray:
    """Prepara la imagen para el modelo Siamés (Normalizada, (1,H,W,1))."""
    try:
        img = Image.open(img_path).convert('L').resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error al preprocesar para modelo: {e}")
        return None

def binarize_for_cv(img_path: str) -> np.ndarray:
    """Prepara la imagen para análisis CV (Binarizada, (H,W), uint8)."""
    try:
        img = Image.open(img_path).convert('L').resize(IMG_SIZE, Image.Resampling.NEAREST)
        img_array = np.array(img, dtype=np.uint8)
        # Umbral simple: 127. Invertido: Fondo=0, Trazo=255 (necesario para contornos internos/agujeros)
        _, bin_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV) 
        return bin_img
    except Exception as e:
        st.error(f"Error al binarizar para CV: {e}")
        return None

# --- 2. Lógica de Carga (Modelo y Plantillas) ---

@st.cache_resource
def load_model_and_templates():
    if not os.path.exists(MODEL_PATH):
        st.error(f"ERROR: Modelo base no encontrado en: {MODEL_PATH}")
        st.stop()
    try:
        base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error al cargar el modelo base: {e}")
        st.stop()

    st.sidebar.info("Cargando plantillas y generando embeddings/binarización...")
    
    template_data = {} # {char_name: {'siam_input': array, 'cv_input': array, 'embedding': array}}
    
    case_dirs = [d for d in os.listdir(TEMPLATE_BASE_DIR) 
                 if os.path.isdir(os.path.join(TEMPLATE_BASE_DIR, d)) and d in ['lower', 'upper', 'numeric']]
    
    # 1. Recopilar nombres y encontrar la mejor plantilla para cada nombre
    for case_dir in case_dirs:
        case_path = os.path.join(TEMPLATE_BASE_DIR, case_dir)
        template_files = [f for f in os.listdir(case_path) 
                          if f.lower().endswith(VALID_IMAGE_EXTENSIONS) and '_template.png' in f]
        
        for template_file in template_files:
            char_name = template_file.replace('_template.png', '')
            
            if char_name not in template_data:
                template_path = os.path.join(case_path, template_file)
                
                siam_input = preprocess_for_siam_model(template_path)
                cv_input = binarize_for_cv(template_path)
                
                if siam_input is not None and cv_input is not None:
                    embedding = base_model.predict(siam_input, verbose=0)[0]
                    
                    template_data[char_name] = {
                        'siam_input': siam_input,
                        'cv_input': cv_input,
                        'embedding': embedding
                    }
    
    final_class_names = sorted(template_data.keys())
    
    if not final_class_names:
        st.error(f"No se pudo cargar ningún dato de plantilla válido para comparación.")
        st.stop()

    template_embeddings = np.array([template_data[name]['embedding'] for name in final_class_names])
    template_cv_inputs = {name: template_data[name]['cv_input'] for name in final_class_names}
    
    st.sidebar.success(f"Embeddings y CV-Templates listos para {len(final_class_names)} clases.")
    
    return base_model, final_class_names, template_embeddings, template_cv_inputs

# --- 3. Lógica de Predicción Principal ---

# Cargar recursos (se ejecuta una vez al inicio)
base_model, template_names, template_embeddings, template_cv_inputs = load_model_and_templates()

st.title("Análisis de Similitud de Caracteres (Siamés vs. Plantilla Específica)")

# --- INTERFAZ DE USUARIO ---

# 1. Selección de la Plantilla Ideal
selected_template_char = st.selectbox(
    "Selecciona la Plantilla Ideal para Comparar:",
    options=['--- Selecciona una Plantilla ---'] + template_names
)

# 2. Carga de la Imagen Escrita a Mano
uploaded_file = st.file_uploader("Sube la imagen de la letra escrita a mano", type=["png", "jpg", "jpeg"])

if selected_template_char != '--- Selecciona una Plantilla ---' and uploaded_file is not None:
    
    # Guardar temporalmente la imagen subida
    temp_file_path = "temp_upload_test.png"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Imagen de Prueba:")
    st.image(uploaded_file, caption="Letra Escrita a Mano", width=150)
    
    # --- A. Preprocesar ---
    user_siam_input = preprocess_for_siam_model(temp_file_path)
    user_cv_bin = binarize_for_cv(temp_file_path)
    
    if user_siam_input is not None and user_cv_bin is not None:
        
        # --- B. Obtener Datos de la Plantilla Seleccionada ---
        template_data_selected = template_cv_inputs.get(selected_template_char)
        template_siam_embedding = template_embeddings[template_names.index(selected_template_char)]
        
        if template_data_selected is None:
            st.error(f"Error: No se encontró la plantilla CV para '{selected_template_char}'")
            st.stop()

        # --- C. Análisis 1: Similitud Siamés (Solo contra la plantilla seleccionada) ---
        with st.spinner("Calculando similitud Siamés..."):
            test_embedding = base_model.predict(user_siam_input, verbose=0)[0]
            
            # Comparar solo con el embedding de la plantilla seleccionada
            siam_distance = np.linalg.norm(template_siam_embedding - test_embedding)
            
            siam_score = max(0.0, 1.0 - (siam_distance / SIMILARITY_THRESHOLD)) * 100
        
        # --- D. Análisis 2: Métricas Geométricas CV (Contra la Plantilla CV Seleccionada) ---
        
        cv_results = {}
        
        with st.spinner("Calculando métricas geométricas CV..."):
            cv_results["inclinacion"] = analyze_inclination(user_cv_bin)
            cv_results["proporcion_wh"] = analyze_proportion(user_cv_bin, template_data_selected)
            cv_results["espaciado"] = analyze_internal_spacing(user_cv_bin, template_data_selected)
            cv_results["consistencia"] = analyze_stroke_consistency(user_cv_bin)
            
            # Generar Feedback (Ajusta la llamada a tu clase/función de feedback)
            feedback_generator = RuleBasedFeedbackGenerator()
            # Puntuación global para el feedback, puedes usar el siamés o un promedio
            deviation_feedback = feedback_generator.generate_feedback(cv_results) 

        # --- E. Mostrar Resultados ---
        st.subheader("Resultado Global (Modelo Siamés)")
        st.info(f"Plantilla Comparada: **{selected_template_char}**")
        st.success(f"Puntuación de Similitud (Siamés, 0-100): **{siam_score:.2f}**")
        st.info(f"Distancia Euclidiana: **{siam_distance:.4f}**")
        
        st.subheader("Diagnóstico Detallado (Comparado con Plantilla Ideal)")
        
        st.metric("Inclinación", f"{cv_results['inclinacion']['score']:.1f}%", delta=f"Ángulo: {cv_results['inclinacion']['user_angle']}°")
        st.metric("Proporción (Ancho/Alto)", f"{cv_results['proporcion_wh']['score']:.1f}%")
        st.metric("Espaciado Interno", f"{cv_results['espaciado']['score']:.1f}%")
        st.metric("Grosor del Trazo", f"{cv_results['consistencia']['score']:.1f}%")
        
        st.subheader("Feedback de Calidad")
        st.write(f"**Fortalezas:** {deviation_feedback.get('fortalezas', 'N/A')}")
        st.write(f"**Áreas a Mejorar:** {deviation_feedback.get('areas_mejora', 'N/A')}")


    # Limpiar archivo temporal
    os.remove(temp_file_path)

else:
    st.info("Por favor, selecciona una plantilla y sube una imagen para comenzar la comparación.")
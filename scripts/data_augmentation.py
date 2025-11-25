"""
Script para generar datos sintéticos (aumentación de plantillas),
separando las variaciones en carpetas 'lower', 'upper' y 'numeric' según la plantilla original.
"""
import os
import cv2
import albumentations as A
import numpy as np
import sys

print("Iniciando la aumentación del dataset...")

# --- CONFIGURACIÓN ---
INPUT_BASE_DIR = "data/templates" # Directorio base que contiene 'lower', 'upper' y 'numeric'
OUTPUT_DIR = "data/variations"
NUM_VARIATIONS_PER_TEMPLATE = 300 # Número de variaciones por imagen base

# Pipeline de aumentación
transform = A.Compose([
    A.Rotate(limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
    A.ElasticTransform(p=0.7, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=0.5),
    A.OpticalDistortion(p=0.5, distort_limit=0.5, shift_limit=0.2),
])


def main():
    """Orquesta la aumentación de datos."""
    
    if not os.path.exists(INPUT_BASE_DIR):
        print(f"Error: El directorio base de plantillas '{INPUT_BASE_DIR}' no existe.")
        print("Asegúrate de ejecutar el script de plantillas primero.")
        return

    # Procesamos 'lower', 'upper' y 'numeric'
    case_types = ['lower', 'upper', 'numeric'] # <-- SE AÑADE 'numeric'
    
    for case_type in case_types:
        input_dir = os.path.join(INPUT_BASE_DIR, case_type)
        
        if not os.path.exists(input_dir):
            print(f"Advertencia: No se encontró el subdirectorio de plantillas: {input_dir}")
            continue
            
        template_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        
        if not template_files:
            print(f"No se encontraron plantillas en '{input_dir}'.")
            continue
            
        print(f"\n--- Procesando {case_type.upper()} ---")
        print(f"Plantillas encontradas: {len(template_files)}. Generando {NUM_VARIATIONS_PER_TEMPLATE} variaciones por plantilla...")

        for template_file in template_files:
            
            # El nombre del carácter base (ej. 'a', 'A', '1', 'Ch')
            raw_character_name = template_file.replace('_template.png', '')

            print(f"Procesando plantilla: {template_file} -> Carácter: {raw_character_name}")
            
            # La estructura de salida será: data/variations/[case_type]/[character_name]/...
            char_output_dir = os.path.join(OUTPUT_DIR, case_type, raw_character_name)
            
            if not os.path.exists(char_output_dir):
                os.makedirs(char_output_dir)
            
            template_path = os.path.join(input_dir, template_file)
            image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None: 
                print(f"Advertencia: No se pudo leer la imagen {template_path}")
                continue

            # Generar y guardar
            for i in range(NUM_VARIATIONS_PER_TEMPLATE):
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                # Nombre del archivo: Ejemplo: a_1.png, 1_50.png, Ch_150.png
                output_filename = f"{raw_character_name}_{i+1}.png"
                output_filepath = os.path.join(char_output_dir, output_filename)
                cv2.imwrite(output_filepath, augmented_image)
        
    print("-" * 30)
    print("¡Aumentación de datos completada!")
    print(f"Imágenes guardadas en: {OUTPUT_DIR}/[lower|upper|numeric]/[character]")
    print("-" * 30)

if __name__ == "__main__":
    main()
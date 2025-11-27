"""
Script para generar las imágenes base de los caracteres (plantillas) separadas por caja.
"""
import os
from PIL import Image, ImageDraw, ImageFont

print("Iniciando la generación de plantillas de caracteres...")

# --- CONFIGURACIÓN ---
BASE_OUTPUT_DIR = "data/templates" # Carpeta principal para las plantillas
IMG_SIZE = (128, 128)
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
FONT_SIZE = 90

# Caracteres divididos por caja y tipo
LOWERCASE_CHARS = list("abcdefghijklmnopqrstuvwxyz") + ['ñ', 'ch', 'll', "rr"]
UPPERCASE_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['Ñ', 'Ch', 'Ll', "Rr"]
NUMBERS = list("0123456789")

# Diccionario que incluye explícitamente las carpetas para cada tipo
ALL_CHARS = {
    'lower': LOWERCASE_CHARS,
    'upper': UPPERCASE_CHARS,
    'numeric': NUMBERS  # <-- SE AÑADE LA CARPETA PARA NÚMEROS
}


# Intenta encontrar una fuente.
try:
    # Nota: Asegúrate de que 'arial.ttf' o una fuente accesible exista en tu sistema o en el directorio del script.
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except IOError:
    print(f"Fuente 'arial.ttf' no encontrada. Usando fuente por defecto.")
    font = ImageFont.load_default()


def generate_character_image(character, filename, output_path):
    """Genera y guarda una imagen para un único caracter."""
    img = Image.new('L', IMG_SIZE, color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Ajuste para fuentes no estándar o sistemas donde textsize está obsoleto
    try:
        # PIL/Pillow >= 9.2.0 usa textbbox
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Versiones anteriores
        text_width, text_height = draw.textsize(character, font=font)
    except Exception:
        # Fallback genérico si hay otros problemas
        text_width, text_height = 50, 50 # Valores aproximados si falla el cálculo

    x = (IMG_SIZE[0] - text_width) / 2
    # Ajuste vertical para centrar mejor (depende de la fuente)
    y = (IMG_SIZE[1] - text_height) / 2 - 5 

    draw.text((x, y), character, fill=TEXT_COLOR, font=font)
    img.save(os.path.join(output_path, filename))


def main():
    """Orquesta la creación de las plantillas separadas por caja."""
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Directorio base creado: {BASE_OUTPUT_DIR}")

    # Generar carpetas de plantillas: 'lower', 'upper', 'numeric'
    for case_type in ALL_CHARS.keys():
        case_output_dir = os.path.join(BASE_OUTPUT_DIR, case_type)
        if not os.path.exists(case_output_dir):
            os.makedirs(case_output_dir)
            print(f"Directorio de plantillas para '{case_type}' creado: {case_output_dir}")

        print(f"\nGenerando plantillas para '{case_type.upper()}'...")
        
        for char in ALL_CHARS[case_type]:
            # El nombre del archivo ahora SÍ incluye la caja/tipo (ej. 'a_template.png', 'A_template.png', '1_template.png')
            output_filename = f"{char}_template.png" 
            
            # Intentamos generar la imagen en su carpeta específica
            generate_character_image(char, output_filename, case_output_dir)

    print("-" * 30)
    print("¡Generación de plantillas completada!")
    print(f"Imágenes guardadas en: {BASE_OUTPUT_DIR}/[lower|upper|numeric]")
    print("-" * 30)


if __name__ == "__main__":
    main()
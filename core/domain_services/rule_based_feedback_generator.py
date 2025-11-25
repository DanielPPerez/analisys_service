"""
Genera feedback para el usuario mapeando los códigos de diagnóstico
a consejos predefinidos basados en reglas.
"""
import random
from typing import Dict, Any

# --- Base de Conocimiento de Consejos ---
CONSEJOS_POR_CODIGO = {
    # ... (Contenido completo del diccionario CONSEJOS_POR_CODIGO) ...
    "proporcion_optima": {
        "tipo": "fortaleza",
        "consejos": [
            "¡Las proporciones de tu letra son excelentes! Muy bien equilibrada.",
            "La relación entre el alto y el ancho de tu letra es perfecta.",
            "Tu letra tiene una forma muy armónica y bien proporcionada."
        ]
    },
    "demasiado_estrecha": {
        "tipo": "mejora",
        "consejos": [
            "Tu letra parece un poco delgada. Intenta darle un poco más de espacio a lo ancho para que respire.",
            "Prueba a hacer el trazo un poco más expansivo. Tu letra está un poco comprimida.",
            "Para mejorar la legibilidad, intenta ensanchar un poco la letra. ¡Vas por buen camino!"
        ]
    },
    "demasiado_ancha": {
        "tipo": "mejora",
        "consejos": [
            "¡Buen trazo! Para la próxima, prueba a hacer la letra un poco más esbelta, no tan ancha.",
            "Tu letra es muy clara, pero un poco ancha. Intenta hacerla un poco más alta que ancha.",
            "Estás ocupando mucho espacio horizontal. Trata de compactar la letra un poco."
        ]
    },
    "optima": {  # Código genérico para proporción óptima
        "tipo": "fortaleza",
        "consejos": [
            "¡Las proporciones de tu letra son excelentes! Muy bien equilibrada.",
            "La relación entre el alto y el ancho de tu letra es perfecta.",
            "Tu letra tiene una forma muy armónica y bien proporcionada."
        ]
    },
    "inclinacion_optima": {
        "tipo": "fortaleza",
        "consejos": [
            "¡Tu postura al escribir es genial! La letra está perfectamente vertical.",
            "La inclinación de tu letra es impecable. ¡Sigue así!",
            "Excelente control; tu letra está muy bien alineada verticalmente."
        ]
    },
    "inclinacion_excesiva_derecha": {
        "tipo": "mejora",
        "consejos": [
            "Tu letra está un poco inclinada hacia la derecha. Intenta mantener tu muñeca y el papel más rectos.",
            "Parece que estás escribiendo un poco rápido. Tómate un segundo para enderezar la letra.",
            "Para mejorar, intenta que el trazo principal de la letra sea más perpendicular a la línea base."
        ]
    },
    "inclinacion_excesiva_izquierda": {
        "tipo": "mejora",
        "consejos": [
            "Tu letra se inclina un poco hacia la izquierda. Asegúrate de que tu mano esté en una posición cómoda y relajada.",
            "Un pequeño ajuste en la postura puede ayudar. Intenta enderezar un poco la letra.",
            "¡Casi perfecto! Prueba a que la letra quede más vertical en lugar de inclinada hacia atrás."
        ]
    },
    # ... (Resto de códigos de INCLINACIÓN) ...
    "variacion_menor": {
        "tipo": "mejora",
        "consejos": [
            "Tu trazo es bastante consistente, pero hay pequeñas variaciones. Intenta mantener una presión más uniforme.",
            "Casi perfecto. Solo necesitas un poco más de uniformidad en el grosor del trazo.",
            "Buen trabajo, pero el trazo podría ser un poco más uniforme en todo su recorrido."
        ]
    },
    "consistencia_optima": {
        "tipo": "fortaleza",
        "consejos": [
            "¡Tu trazo es muy firme y consistente! La presión que aplicas es muy uniforme.",
            "Excelente control del lápiz. El grosor de tu letra es muy regular.",
            "La consistencia de tu trazo es de libro. ¡Muy buen trabajo!"
        ]
    },
    "trazo_inconsistente": {
        "tipo": "mejora",
        "consejos": [
            "Intenta mantener una presión más constante al escribir. Algunas partes del trazo son más gruesas que otras.",
            "Tu trazo varía un poco. Concéntrate en hacer un movimiento fluido y con la misma presión.",
            "Para un acabado más limpio, prueba a que el grosor de la línea no cambie tanto de principio a fin."
        ]
    },
    # ... (Resto de códigos de CONSISTENCIA) ...
    "espaciado_interno_optimo": {
        "tipo": "fortaleza",
        "consejos": [
            "¡El espacio dentro de tu letra está perfectamente definido! Muy legible.",
            "Los bucles y círculos de tu letra tienen un tamaño ideal. ¡Excelente!",
            "Muy buen trabajo al definir los espacios internos de la letra."
        ]
    },
    "circulo_casi_cerrado": {
        "tipo": "mejora",
        "consejos": [
            "Casi lo tienes. El círculo de tu letra está un poco aplastado. Dale más aire para que respire.",
            "¡A un paso de la perfección! Intenta abrir un poco más el espacio interior de la letra.",
            "Para que sea más fácil de leer, asegúrate de que los bucles internos no se cierren por completo."
        ]
    },
    "circulo_demasiado_abierto": {
        "tipo": "mejora",
        "consejos": [
            "¡Buen trabajo! Para la próxima, intenta cerrar un poco más el círculo o el bucle de la letra.",
            "Tu letra es clara, pero el espacio interior es muy grande. Prueba a hacerlo un poco más pequeño.",
            "El trazo está bien, pero no termines de cerrar la forma. Intenta que los extremos se unan un poco más."
        ]
    },
    "wrong_hole_count": {
        "tipo": "mejora",
        "consejos": [
            "Parece que la forma básica de la letra no es correcta. Por ejemplo, una 'o' se convirtió en una 'u'.",
            "Revisa bien la estructura de la letra. El número de espacios cerrados no coincide con la plantilla.",
            "¡Ojo! Una letra como la 'B' debe tener dos espacios cerrados, y una 'P' solo uno. ¡Revisa tu trazo!"
        ]
    },
    "no_holes_expected": {
        "tipo": "mejora",
        "consejos": [
            "Tu trazo ha creado un círculo donde no debería haberlo. Por ejemplo, una 'u' que parece una 'o'.",
            "Asegúrate de no cerrar completamente la letra si no es necesario.",
            "Esta letra no debería tener espacios internos cerrados. Revisa el modelo y tu trazo."
        ]
    },
    "no_contour_found": {
        "tipo": "error",
        "consejos": [
            "No pudimos detectar una letra en la imagen. Asegúrate de escribir dentro del área designada.",
            "La imagen parece estar en blanco o el trazo es demasiado tenue. Intenta escribir con más claridad.",
        ]
    },
    "not_enough_data": {
        "tipo": "error",
        "consejos": [
            "El trazo que hiciste es muy pequeño o corto para poder analizarlo en detalle. Intenta hacerlo un poco más grande.",
            "Necesitamos un trazo un poco más largo para poder darte un buen consejo sobre este punto."
        ]
    }
}

class RuleBasedFeedbackGenerator:
    """Genera feedback para el usuario mapeando los códigos de diagnóstico a consejos."""
    def generate_feedback(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Analiza las métricas y genera un diccionario con fortalezas y áreas de mejora."""
        strengths = []
        improvements = []
        
        # Mapeo de nombres de métricas a nombres más legibles
        metric_names = {
            "inclinacion": "Inclinación",
            "proporcion": "Proporción",
            "proporcion_wh": "Proporción",  # Alias
            "espaciado": "Espaciado interno",
            "consistencia": "Consistencia del trazo"
        }

        for metric_name, details in metrics.items():
            if not isinstance(details, dict): 
                continue
            
            code = details.get("deviation_code")
            score = details.get("score", 0)
            
            # Saltar métricas sin código o con errores
            if not code or code in ["no_template", "error", "no_contour_found", "no_content", "no_data"]:
                continue
            
            # Mapear código genérico "optima" a códigos específicos según la métrica
            if code == "optima":
                if metric_name == "proporcion":
                    code = "proporcion_optima"
                elif metric_name == "espaciado":
                    code = "espaciado_interno_optimo"
                elif metric_name == "consistencia":
                    code = "consistencia_optima"
                # Para inclinación, ya tiene su código específico
            
            if code in CONSEJOS_POR_CODIGO:
                info = CONSEJOS_POR_CODIGO[code]
                consejo = random.choice(info["consejos"])
                
                # Usar puntuación real para decidir si es fortaleza o mejora
                if score >= 90 and info["tipo"] != "error":
                    # Alta puntuación = fortaleza
                    metric_display = metric_names.get(metric_name, metric_name)
                    strengths.append(f"{metric_display}: {consejo}")
                elif score < 70 and info["tipo"] == "mejora":
                    # Baja puntuación = área de mejora importante
                    metric_display = metric_names.get(metric_name, metric_name)
                    improvements.append(f"{metric_display}: {consejo}")
                elif info["tipo"] == "mejora" and score < 90:
                    # Mejora moderada
                    metric_display = metric_names.get(metric_name, metric_name)
                    improvements.append(f"{metric_display}: {consejo}")
        
        # Generar mensaje final de fortalezas
        if strengths:
            if len(strengths) == 1:
                final_strength = strengths[0]
            elif len(strengths) >= 2:
                final_strength = f"{strengths[0]} Además, {strengths[1].lower()}"
            else:
                final_strength = ". ".join(strengths[:2])
        else:
            final_strength = "¡Sigue practicando, vas por muy buen camino!"
        
        # Generar mensaje final de mejoras
        if improvements:
            # Priorizar las mejoras más importantes (menor puntuación)
            if len(improvements) == 1:
                final_improvement = improvements[0]
            else:
                # Combinar las dos mejoras más importantes
                final_improvement = f"{improvements[0]} También, {improvements[1].lower()}"
        elif not strengths:
            final_improvement = "Parece que la imagen está en blanco o el trazo es muy tenue. ¡Inténtalo de nuevo!"
        else:
            final_improvement = "¡Tu letra es prácticamente perfecta! No tenemos ninguna sugerencia por ahora."

        return {
            "fortalezas": final_strength,
            "areas_mejora": final_improvement
        }
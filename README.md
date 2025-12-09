# Analisys-service

Servicio de análisis de caligrafía usando modelos de Machine Learning (Siamese Networks) y análisis geométrico con OpenCV.

## Estructura del Proyecto

- `core/`: Lógica de dominio, modelos ML, servicios de análisis
- `application/`: Casos de uso
- `infrastructure/`: Adaptadores (ML, LLM, API)
- `scripts/`: Scripts de entrenamiento y utilidades
- `data/`: Datos, modelos entrenados y plantillas
- `tests/`: Pruebas unitarias e integración

## Requisitos Previos

- Python 3.8+
- Git LFS (para descargar los modelos)

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Analisys-service
```

### 2. Instalar Git LFS (si no está instalado)

**Windows:**
```bash
# Descargar desde https://git-lfs.github.com/
# O usar chocolatey:
choco install git-lfs
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# macOS
brew install git-lfs
```

### 3. Inicializar Git LFS y descargar los modelos

```bash
git lfs install
git lfs pull
```

**Nota importante:** Los modelos entrenados (`base_model.keras` y `siamese_model.keras`) se almacenan usando Git LFS debido a su tamaño. Asegúrate de tener Git LFS instalado y ejecutar `git lfs pull` después de clonar para descargar los archivos completos.

### 4. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Uso

### Entrenar el modelo

```bash
python scripts/train.py
```

### Ejecutar la API

```bash
python infrastructure/api/v1/main.py
```

## Modelos Entrenados

Los modelos pre-entrenados se encuentran en `data/models/`:
- `base_model.keras`: Red base para extracción de características
- `siamese_model.keras`: Modelo Siamese completo

Estos archivos se gestionan con Git LFS. Si al clonar el repositorio los archivos aparecen vacíos o muy pequeños, ejecuta:
```bash
git lfs pull
```

## Estructura de Datos

- `data/templates/`: Plantillas de referencia para cada carácter
- `data/variations/`: Variaciones de entrenamiento
- `data/models/`: Modelos entrenados (Git LFS)

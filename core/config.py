import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class Settings:
    """Clase para centralizar todas las configuraciones del proyecto."""
    API_KEY_OPENAI: str = os.getenv("API_KEY_OPENAI")

# Instancia Singleton
settings = Settings()
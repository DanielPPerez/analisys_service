# infrastructure/api/v1/main.py
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from infrastructure.api.v1.endpoints import analysis_endpoint

def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI app."""
    app = FastAPI(
        title="Analysis Service API",
        description="API para el análisis de caligrafía y generación de feedback.",
        version="1.0.0"
    )

    # Middleware de seguridad (MSTG-NETWORK-1)
    # Solo acepta requests del API Gateway o hosts confiables
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.scriptoria.com", "localhost", "analysis-service", "127.0.0.1"]
    )

    # Incluir los endpoints de análisis
    app.include_router(analysis_endpoint.router, prefix="/v1", tags=["Analysis"])

    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "Analysis Service API Running"}
    
    return app

# Crear la instancia de la app para compatibilidad con uvicorn directo
app = create_app()

# NOTA: Para correr esto, usa uvicorn: uvicorn infrastructure.api.v1.main:app --reload
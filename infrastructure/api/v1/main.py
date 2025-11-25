# infrastructure/api/v1/main.py
from fastapi import FastAPI
from .endpoints import analysis_endpoint

app = FastAPI(
    title="Analysis Service API",
    description="API para el análisis de caligrafía y generación de feedback.",
    version="1.0.0"
)

# Incluir los endpoints de análisis
app.include_router(analysis_endpoint.router, prefix="/v1", tags=["Analysis"])

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Analysis Service API Running"}

# NOTA: Para correr esto, usa uvicorn: uvicorn infrastructure.api.v1.main:app --reload
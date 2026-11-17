"""API REST para clasificar reseñas de restaurantes con el pipeline entrenado."""

from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_nlp.pkl"

app = FastAPI(
    title="API de análisis de sentimientos",
    description="Clasifica reseñas de restaurantes como positivas o negativas.",
    version="1.0.0",
)

# Cargar el pipeline una vez al iniciar el servicio.
modelo = joblib.load(MODEL_PATH)


class ResenaInput(BaseModel):
    review: str = Field(..., description="Texto de la reseña", min_length=1)

    @field_validator("review")
    @classmethod
    def validar_resena_no_vacia(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La reseña no puede estar vacía.")
        return value


@app.get("/health")
def health() -> dict[str, str]:
    """Indica que la API está lista y que el modelo se cargó correctamente."""
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predecir_sentimiento(datos: ResenaInput) -> dict[str, str | int]:
    """Predice el sentimiento de una reseña."""
    prediccion = int(modelo.predict([datos.review])[0])
    sentimiento = "Positiva" if prediccion == 1 else "Negativa"
    return {
        "review": datos.review,
        "liked": prediccion,
        "sentimiento": sentimiento,
    }

# API de análisis de sentimientos

La API carga `modelo_nlp.pkl`, generado por `train.py` usando el vectorizador con mayor F1 macro entre Bag-of-Words y TF-IDF.

## Entrenar el pipeline

```powershell
python train.py
```

## Iniciar el servicio

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: <http://127.0.0.1:8000/docs>

## Consultar una predicción

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"review":"The food was amazing and the service was excellent!"}'
```

`liked` vale `1` para una reseña positiva y `0` para una negativa.

# Checkpoint 2 — Clasificación de reseñas con vectorización

Este proyecto amplía el clasificador de reseñas de restaurantes y compara las técnicas de vectorización vistas en clase: Bag-of-Words (BoW) y TF-IDF.

## Archivos

- `Restaurant_Reviews.tsv`: reseñas etiquetadas; `Liked` vale 1 para positiva y 0 para negativa.
- `train.py`: separa entrenamiento/prueba, ajusta ambos vectorizadores con el mismo clasificador y selecciona el de mejor F1 macro.
- `resultados_vectorizacion.json`: métricas de accuracy, F1 macro y reporte por clase para ambas opciones.
- `modelo_nlp.pkl`: pipeline del vectorizador ganador, listo para la API.
- `app.py`: API REST que usa ese mismo pipeline en reseñas nuevas.
- `requirements.txt`: dependencias congeladas del entorno virtual.

## Técnicas comparadas

- **BoW (`CountVectorizer`)** cuenta términos y conserva su frecuencia; es simple de interpretar, pero no reduce el peso de términos frecuentes en todo el corpus.
- **TF-IDF (`TfidfVectorizer`)** pondera los términos según su frecuencia en la reseña y su rareza en el corpus, dando más peso a términos distintivos.

Ambos modelos usan una división estratificada con semilla fija y regresión logística. El mejor F1 macro determina el pipeline guardado en `modelo_nlp.pkl`.

## Preparación y ejecución (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

Para iniciar la API después de entrenar:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000/docs` para consultar la API. El rendimiento reportado corresponde a la partición de prueba de este conjunto de datos.

"""Compara Bag-of-Words y TF-IDF para clasificar reseñas."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Restaurant_Reviews.tsv"
MODEL_PATH = BASE_DIR / "modelo_nlp.pkl"
RESULTS_PATH = BASE_DIR / "resultados_vectorizacion.json"


def create_pipeline(vectorizer) -> Pipeline:
    """Une el vectorizador con el clasificador."""
    return Pipeline([
        ("vectorizacion", vectorizer),
        ("clasificador", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def main() -> None:
    reviews = pd.read_csv(DATA_PATH, sep="\t", quoting=3)
    X_train, X_test, y_train, y_test = train_test_split(
        reviews["Review"].astype(str),
        reviews["Liked"],
        test_size=0.2,
        random_state=42,
        stratify=reviews["Liked"],
    )

    candidates = {
        "Bag-of-Words (CountVectorizer)": CountVectorizer(
            lowercase=True, stop_words="english"
        ),
        "TF-IDF (TfidfVectorizer)": TfidfVectorizer(
            lowercase=True, stop_words="english"
        ),
    }

    results = {}
    fitted_models = {}
    for name, vectorizer in candidates.items():
        pipeline = create_pipeline(vectorizer)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        }
        fitted_models[name] = pipeline

    selected_name = max(results, key=lambda name: results[name]["f1_macro"])
    joblib.dump(fitted_models[selected_name], MODEL_PATH)
    output = {
        "dataset": DATA_PATH.name,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "selected_vectorizer": selected_name,
        "models": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Reseñas de entrenamiento: {len(X_train)}")
    print(f"Reseñas de prueba: {len(X_test)}")
    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f}, "
            f"F1 macro={metrics['f1_macro']:.3f}"
        )
    print(f"Vectorizador seleccionado: {selected_name}")
    print(f"Pipeline guardado: {MODEL_PATH}")
    print(f"Métricas guardadas: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

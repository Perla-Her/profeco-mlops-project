import unicodedata
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

RUTA_DEFAULT = "models/recomendador.pkl"


def _normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).strip()


def construir_catalogo(df):
    df = df.copy()
    df["marca"] = df["marca"].fillna("")
    df["texto"] = (df["producto"].fillna("") + " " + df["marca"]).map(_normalizar)
    return (
        df.groupby(["texto", "producto", "marca", "unidad_base", "catalogo"],
                   as_index=False, dropna=False)
        .agg(precio_unitario=("precio_unitario", "median"),
             n_registros=("precio_unitario", "size"))
    )


def _evaluar(cat, matriz, nn, k, n_muestra=300, semilla=42):
    """Precision@k: % de vecinos que son del mismo catálogo que el producto base."""
    pos = np.random.RandomState(semilla).choice(
        len(cat), min(n_muestra, len(cat)), replace=False)
    _, idx = nn.kneighbors(matriz[pos], n_neighbors=min(k + 1, len(cat)))
    aciertos = []
    for fila, vecinos in zip(pos, idx):
        vecinos = [v for v in vecinos if v != fila][:k]
        aciertos.append(
            (cat["catalogo"].iloc[vecinos] == cat["catalogo"].iloc[fila]).mean())
    return float(np.mean(aciertos))


def entrenar_recomendador(df, config):
    k = config.get("top_k", 5)
    cat = construir_catalogo(df)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    matriz = vec.fit_transform(cat["texto"])
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(matriz)

    ruta = Path(config.get("recommender_path", RUTA_DEFAULT))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vec": vec, "nn": nn, "catalogo": cat}, ruta)
    return {
        "n_productos": len(cat),
        f"precision_at_{k}_catalogo": round(_evaluar(cat, matriz, nn, k), 4),
        "recommender_path": str(ruta),
    }


def recomendar(consulta, k=5, ruta=RUTA_DEFAULT, candidatos=50, min_similitud=0.3):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el recomendador en '{ruta}'. "
            "Ejecuta primero: python -m src.main")
    art = joblib.load(ruta)
    cat = art["catalogo"]
    q = art["vec"].transform([_normalizar(consulta)])
    dist, idx = art["nn"].kneighbors(q, n_neighbors=min(candidatos, len(cat)))
    cand = cat.iloc[idx[0]].copy()
    cand["similitud"] = 1 - dist[0]

    ref = cand.iloc[0]
    if ref["similitud"] < 0.1:
        raise ValueError(f"No encontré ningún producto parecido a '{consulta}'.")

    sust = cand.iloc[1:]
    sust = sust[
        (sust["unidad_base"] == ref["unidad_base"])
        & (sust["precio_unitario"] < ref["precio_unitario"])
        & (sust["similitud"] >= min_similitud)
    ].sort_values("precio_unitario").head(k).copy()
    sust["ahorro_pct"] = (1 - sust["precio_unitario"] / ref["precio_unitario"]) * 100
    return ref, sust
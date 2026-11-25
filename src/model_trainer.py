from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_and_save_model(X_train, y_train, X_test, y_test, config=None):
    """
    Entrena y compara modelos de regresión, guarda el mejor pipeline en models/
    y retorna un diccionario con las métricas del modelo ganador y la comparativa.
    """
    config = config or {}
    random_state = config.get("random_state", 42)
    ruta_modelo = Path(config.get("model_path", "models/mejor_modelo.pkl"))

    # 1. Preprocesamiento
    otras_categoricas = [c for c in X_train.columns if c != "calidad"]

    transformadores = []
    if "calidad" in X_train.columns:
        calidad_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="estandar")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformadores.append(("calidad", calidad_pipe, ["calidad"]))

    if otras_categoricas:
        otras_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformadores.append(("cat", otras_pipe, otras_categoricas))

    preprocessor = ColumnTransformer(transformers=transformadores)
    return preprocessor
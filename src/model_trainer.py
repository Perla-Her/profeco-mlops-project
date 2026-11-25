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

    #Preprocesamiento
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

    modelos = {
        "Regresion_Lineal": LinearRegression(),
        "Ridge": Ridge(alpha=config.get("ridge_alpha", 1.0)),
        "Random_Forest": RandomForestRegressor(
            n_estimators=config.get("rf_n_estimators", 50),
            max_depth=config.get("rf_max_depth", 20),
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=config.get("gb_n_estimators", 100),
            random_state=random_state,
        ),
    }

    mejor_modelo = None
    mejor_rmse = float("inf")
    mejor_nombre = ""
    resultados = {}

    print("Iniciando entrenamiento y comparación de modelos...\n")

    for nombre, modelo in modelos.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", modelo),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))

        resultados[nombre] = {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }

        print(f"Modelo: {nombre:18} | R2: {r2:.4f} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f}")

        if rmse < mejor_rmse:
            mejor_rmse = rmse
            mejor_modelo = pipeline
            mejor_nombre = nombre

    return resultados

    print(f"\nEl mejor modelo fue {mejor_nombre} con un RMSE de ${mejor_rmse:.2f}")

    #Guardar el mejor modelo
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor_modelo, ruta_modelo)
    print(f"Modelo guardado exitosamente en '{ruta_modelo}'")

    #Diccionario de métricas
    return {
        "mejor_modelo": mejor_nombre,
        "r2": resultados[mejor_nombre]["r2"],
        "mae": resultados[mejor_nombre]["mae"],
        "rmse": resultados[mejor_nombre]["rmse"],
        "model_path": str(ruta_modelo),
        "detalle_modelos": resultados,
    }

#Pruebas locales
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data_loader import preparar_datos

    df = preparar_datos("data/raw/proyecto.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["precio_unitario"])

    features = ["producto", "marca", "estado", "nombre_comercial", "unidad_base", "calidad"]
    X = df[features]
    y = df["precio_unitario"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metricas = train_and_save_model(X_train, y_train, X_test, y_test, config={})
    print("\nDiccionario retornado:", metricas)

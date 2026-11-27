import yaml

from src.data_loader import cargar_catalogo, dividir_datos
from src.model_trainer import train_and_save_model
from src.recommender import entrenar_recomendador


def cargar_config(ruta="config/params.yaml"):
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = cargar_config()

    print("1/3 Cargando y limpiando datos...")
    df = cargar_catalogo(config)
    X_train, X_test, y_train, y_test = dividir_datos(df, config)

    print("2/3 Entrenando regresores (precio unitario esperado)...")
    metricas_reg = train_and_save_model(X_train, y_train, X_test, y_test, config)

    print("3/3 Construyendo recomendador de sustitutos...")
    metricas_rec = entrenar_recomendador(df, config)

    print("\nRegresión:", metricas_reg)
    print("Recomendador:", metricas_rec)


if __name__ == "__main__":
    main()
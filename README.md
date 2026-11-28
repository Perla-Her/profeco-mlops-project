# Profeco MLOps Project

Recomendador de **sustitutos más baratos** para productos de la base "Quién es quién en los precios (Julio 2026 parte 1)" de Profeco. Dado un producto, el sistema encuentra productos similares (misma unidad de medida) con menor precio unitario y calcula el ahorro estimado.

El proyecto sigue una estructura MLOps con un rol por módulo: carga de datos, entrenamiento, orquestación y predicción.

## Cómo funciona

1. **Limpieza** (`data_loader.py`): carga el CSV, quita Medicamentos y Electrodomésticos, extrae cantidad y unidad de la presentación, calcula el precio unitario (por kg o litro) y detecta la calidad (integral, extra, super extra).
2. **Regresores** (`model_trainer.py`): entrena y compara 4 modelos (Regresión Lineal, Ridge, Random Forest y Gradient Boosting) para estimar el precio unitario esperado. Guarda el mejor (menor RMSE) en `models/mejor_modelo.pkl`.
3. **Recomendador** (`recommender.py`): representa cada producto (nombre y marca) con TF-IDF y busca vecinos con similitud coseno. Recomienda los más parecidos que tengan la misma unidad y un precio unitario menor. Se guarda en `models/recomendador.pkl`.
4. **Orquestación** (`main.py`): ejecuta todo en orden leyendo `config/params.yaml`.
5. **Predicción** (`predict.py`): consulta el recomendador desde la terminal.

## Estructura

```
profeco-mlops-project/
├── config/
│   └── params.yaml        # Parámetros de datos, modelos y recomendador
├── data/
│   ├── raw/               # CSV original (no se sube a Git)
│   └── processed/
├── models/                # Modelos .pkl generados (no se suben a Git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Rol: Data Engineer
│   ├── model_trainer.py   # Rol: ML Engineer
│   ├── recommender.py     # Recomendador de sustitutos
│   ├── main.py            # Rol: MLOps Engineer
│   └── predict.py         # Rol: QA & Production Engineer
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requisitos

- Python 3.11 o superior
- Las dependencias de `requirements.txt`

## Instalación

```bash
git clone https://github.com/Perla-Her/profeco-mlops-project.git
cd profeco-mlops-project
python -m venv .venv
source .venv/Scripts/activate    # En Git Bash (Windows). En Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Los datos no se incluyen en el repositorio. Descarga la base "Quién es quién en los precios" de Profeco ([https://www.datos.gob.mx/dataset/programa_quien_es_quien_precios_2026/resource/40f64926-bd6b-4642-8e04-e767d9775d4d]) y guárdala como:

```
data/raw/proyecto.csv
```

Columnas que usa el proyecto: `producto`, `marca`, `estado`, `nombre_comercial`, `presentacion`, `precio` y `catalogo`. Si el archivo tiene otro nombre o ruta, cambia `data_path` en `config/params.yaml`.

## Uso

Entrenar todo el pipeline (desde la raíz del proyecto):

```bash
python -m src.main
```

Pedir sustitutos para un producto:

```bash
python -m src.predict "frijol negro"
python -m src.predict "aceite" 3     # el segundo argumento es cuántos sustitutos mostrar
```

Ejemplo de salida:

```
Producto de referencia: Frijol (S/M) - $31.00 por kg

Sustitutos más baratos:
- Frijol (S/M): $20.00 (ahorro 35%)
- Frijol (El Labrador): $21.00 (ahorro 32%)
```

Si el modelo no existe, `predict` pide ejecutar antes `python -m src.main`.

## Configuración

Todos los parámetros están en `config/params.yaml`:

| Parámetro | Descripción |
|---|---|
| `data_path` | Ruta del CSV |
| `sample_size` | Máximo de filas a usar (`null` para todo el dataset) |
| `test_size`, `random_state` | División train/test y semilla |
| `ridge_alpha`, `rf_n_estimators`, `rf_max_depth`, `gb_n_estimators` | Hiperparámetros de los regresores |
| `model_path`, `recommender_path` | Dónde se guardan los modelos |
| `top_k` | Número de recomendaciones |

## Resultados

Regresores (precio unitario, conjunto de prueba):

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| Regresión Lineal | 0.9415 | $33.81 | $80.97 |
| **Ridge** | **0.9435** | **$33.64** | **$79.53** |
| Random Forest | 0.8604 | $84.01 | $125.04 |
| Gradient Boosting | 0.8433 | $92.46 | $132.50 |

Recomendador: 1,168 productos en el catálogo y Precision@5 (por catálogo) de 0.50, es decir, la mitad de los vecinos recomendados pertenece a la misma categoría que el producto de referencia.

## Equipo

| Rol | Responsable |
|---|---|
| Data Engineer | [Samuel Salvador Cintora Ruvalcaba] |
| ML Engineer | [Isaac Emanuel Molotla Jimenez] |
| MLOps Engineer | [Perla Hernandez Juarez] |
| QA & Production Engineer | [Kitzia Denisse Carretero Contreras] |

## Flujo de trabajo

Una rama por cambio (`feature/...`), Pull Request revisado por otra persona hacia `main`, y eliminación de la rama después del merge.
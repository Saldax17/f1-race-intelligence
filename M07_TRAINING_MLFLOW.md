# M7 — entrenamiento y MLflow

## Alcance e invariantes

M7 consume exclusivamente un reporte M6 con veredicto `VALID` o
`VALID_WITH_WARNINGS`. Antes de entrenar ejecuta la verificación existente de
M6 sobre el reporte y los ocho artefactos declarados, valida el target
`next_lap_time` y usa en el orden de M6 `preprocessing.output_feature_names`.
No vuelve a seleccionar columnas, recalcular features, ajustar el preprocesador
ni crear splits. El target y las columnas de trazabilidad no se envían al
estimador. Un error de integridad, un reporte no válido o un backend distinto
de `local` detiene el stage.

M7 soporta dos familias de modelos seleccionables mediante
`training.model_type` (o el flag CLI `--model-type`):

- **`linear_regression` (por defecto):** entrena
  `sklearn.linear_model.LinearRegression` como modelo principal (registrado vía
  `mlflow.sklearn.log_model` bajo `f1-next-lap-linear-regression`) junto a la
  referencia `DummyRegressor(strategy="mean")`, ambos ajustados solo con
  `train`. El comportamiento baseline original permanece intacto.
- **`mlp` (Deep Learning):** entrena un MLP tabular en PyTorch CPU-only
  (`f1_race_intelligence.modeling.dl.TabularMLP`, registrado vía
  `mlflow.pyfunc.log_model` bajo `f1-next-lap-mlp`), con estandarización del
  target ajustada solo en `train` (inversa embebida en el modelo) y *early
  stopping* sobre `validation`. En el mismo run ajusta también
  `LinearRegression` y `DummyRegressor(strategy="mean")` sobre `train` para
  comparación directa sobre el mismo reporte M6. Requiere el extra opcional `dl`
  (`requirements/dl-lock.txt`); ver [M07_DEEP_LEARNING.md](M07_DEEP_LEARNING.md)
  para arquitectura, hiperparámetros, anti-leakage y portabilidad.

Se calculan MAE, RMSE, MAPE y R² en train, validation y test. MAPE excluye
targets cero y devuelve `null` cuando no hay ningún denominador válido. Los
targets y predicciones no finitos se excluyen y se cuentan; R² se informa como
`null` cuando no está definido.

## Trazabilidad e inferencia

Cada experimento registra `model_type`, `estimator`, el `run_id`/versión del
reporte M6, versión del pipeline, `input_fingerprint`, hashes verificados de
artefactos, commits M6/M7, configuración, seed, métricas y reporte M6 original.
También adjunta el preprocesador y el esquema M6 (y en corridas `mlp`, las
curvas de pérdida por época y `training_history.json`). El modelo registrado
recibe la matriz numérica codificada de M6, en el orden de
`output_feature_names`; M7 carga el URI del registro al terminar vía
`mlflow.pyfunc.load_model` y comprueba que la predicción coincida con la del
modelo recién ajustado (`rtol`/`atol` de `1e-10` para `linear_regression` y
`1e-5` para `mlp`).

Tracking y Model Registry usan el mismo backend local SQLite. El extra
`training` fija `mlflow==3.18.0` y `SQLAlchemy==2.0.57`, compatibles con las
versiones pandas/pyarrow del runtime-lock; el extra opcional `dl` fija
`torch==2.12.1` (wheels CPU-only en `requirements/dl-lock.txt` para Linux y
Windows, y macOS Apple Silicon ≥ 14 vía wheel `macosx_14_0_arm64`; en Mac Intel
o macOS < 14 no existe wheel precompilado de `torch 2.12.1`) y se importa de
forma perezosa para mantener aislado el runtime base M1–M6. El directorio de
artefactos es local y separado del archivo SQLite:

- Tracking/Registry: `data/mlflow/tracking.db`
- Artefactos MLflow: `data/mlflow/artifacts/`
- Reportes de M7: `data/modeling/training/reports/`

Los valores se pueden cambiar en la sección `training` de `configs/config.yaml`
(incluyendo `training.model_type` y la subsección `training.mlp`).
No se inicializa un servidor, no se usa un tracking URI remoto y no se usa AWS.
El experimento existente debe conservar su `artifact_path`; si se cambia, se
debe configurar otra base SQLite o restaurar la ruta anterior.

## Ejecución

Instalar las dependencias del proyecto para el baseline lineal (`MLflow` está
fijado en `pyproject.toml`):

```powershell
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -r requirements/dev-lock.txt
python -m pip install -e ".[dev,training]" --no-deps
```

Para habilitar también `--model-type mlp`, instalar adicionalmente
`requirements/dl-lock.txt` y activar el extra `dl` (ver
[M07_DEEP_LEARNING.md](M07_DEEP_LEARNING.md)):

```powershell
python -m pip install -r requirements/dl-lock.txt
python -m pip install -e ".[dev,training,dl]" --no-deps
```

Con un reporte M6 ya producido:

```powershell
# Baseline lineal por defecto (model_type=linear_regression)
f1-pipeline --stage m7 --m6-report data/modeling/reports/modeling_report_<timestamp>_<run-id>.json

# Deep Learning tabular (PyTorch CPU-only MLP)
f1-pipeline --stage m7 --model-type mlp --m6-report data/modeling/reports/modeling_report_<timestamp>_<run-id>.json
```

`--m6-report` también se puede establecer como `training.m6_report_path`,
`--model-type` sobrescribe `training.model_type` y `--registered-model-name`
permite fijar el nombre del modelo en el Model Registry para esa ejecución.
Para examinar tracking y modelos localmente:

```powershell
mlflow ui --backend-store-uri sqlite:///data/mlflow/tracking.db
```

Recuperación para inferencia (baseline lineal con `mlflow.sklearn`, o de forma
uniforme para `linear_regression` y `mlp` mediante `mlflow.pyfunc.load_model`):

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///data/mlflow/tracking.db")
model = mlflow.sklearn.load_model("models:/f1-next-lap-linear-regression/<version>")
predictions = model.predict(m6_encoded_features[encoded_feature_names])
```

`m6_encoded_features` se obtiene aplicando el `preprocessor.joblib` M6 verificado
a las columnas de entrada aprobadas por M6; `encoded_feature_names` proviene
del `feature_schema.json` de esa misma ejecución. El URI y la versión concreta
también están en el reporte M7 y en los tags del run. Para empaquetar de forma
inmutable una versión registrada (lineal o MLP) y servirla en M8, ver
[M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md).

## Archivos de M7

- `src/f1_race_intelligence/modeling/training.py`: verificación M6, carga de
  matrices M6, dispatch por `model_type` (`linear_regression` y `mlp`),
  baselines, métricas, registro, comprobación de recuperación y reporte M7.
- `src/f1_race_intelligence/modeling/dl.py`: implementación del MLP tabular en
  PyTorch CPU-only, escalado anti-leakage del target, checkpoint
  `weights_only=True` y wrapper `mlflow.pyfunc.PythonModel`.
- `src/f1_race_intelligence/config/settings.py` y `configs/config.yaml`:
  configuración local M7 (`training`, `training.model_type` y `training.mlp`),
  aditiva a las secciones existentes.
- `src/f1_race_intelligence/cli.py`: stage `m7` y opciones `--m6-report`,
  `--model-type` y `--registered-model-name`.
- `pyproject.toml`, `requirements/training-lock.txt` y
  `requirements/dl-lock.txt`: extras opcionales `training` y `dl` fijados y
  aislados del runtime base M1–M6.
- `tests/unit/test_training.py`, `tests/unit/test_dl.py` y
  `tests/unit/test_dl_m7_m8_integration.py`: métricas con ceros/nulos, flujo
  completo M7 (lineal y MLP) con fixtures sintéticos: reproducibilidad,
  anti-leakage, tracking, registro, publicación inmutable y serving M8.
- `M07_DEEP_LEARNING.md` y `README.md`: documentación técnica y alcance de M7.

No se cambian splits, contratos, features ni guardas M1–M6.

## Validación y estado

Las pruebas usan datasets sintéticos con forma M5 y artefactos M6 generados en
directorios temporales. Las métricas resultantes son solo comprobaciones
técnicas: **no son resultados académicos ni describen carreras reales**.
Ninguna prueba consulta OpenF1, descarga datos o ejecuta entrenamiento sobre
datos reales.

Pendiente: ejecutar M6/M7 (tanto `linear_regression` como `mlp`) sobre los
datos reales únicamente con autorización, revisar suficiencia de carreras y
discutir los resultados con el equipo. LSTM/GRU y métodos estadísticos
avanzados quedan fuera de alcance.

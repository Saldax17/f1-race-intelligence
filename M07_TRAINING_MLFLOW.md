# M7.1 — entrenamiento y MLflow

## Alcance e invariantes

M7 consume exclusivamente un reporte M6 con veredicto `VALID` o
`VALID_WITH_WARNINGS`. Antes de entrenar ejecuta la verificación existente de
M6 sobre el reporte y los ocho artefactos declarados, valida el target
`next_lap_time` y usa en el orden de M6 `preprocessing.output_feature_names`.
No vuelve a seleccionar columnas, recalcular features, ajustar el preprocesador
ni crear splits. El target y las columnas de trazabilidad no se envían al
estimador. Un error de integridad, un reporte no válido o un backend distinto
de `local` detiene el stage.

El modelo principal es `sklearn.linear_model.LinearRegression`; la referencia
es `DummyRegressor(strategy="mean")`, ambos ajustados solo con `train`. Se
calculan MAE, RMSE, MAPE y R² en train, validation y test. MAPE excluye targets
cero y devuelve `null` cuando no hay ningún denominador válido. Los targets y
predicciones no finitos se excluyen y se cuentan; R² se informa como `null`
cuando no está definido.

## Trazabilidad e inferencia

Cada experimento registra el `run_id`/versión del reporte M6, versión del
pipeline, `input_fingerprint`, hashes verificados de artefactos, commits M6/M7,
configuración, seed, métricas y reporte M6 original. También adjunta el
preprocesador y el esquema M6. El modelo registrado recibe la matriz numérica
codificada de M6, en el orden de `output_feature_names`; M7 carga el URI del
registro al terminar y comprueba que la predicción coincida con la del modelo
recién ajustado.

Tracking y Model Registry usan el mismo backend local SQLite. El extra
`training` fija `mlflow==3.18.0` y `SQLAlchemy==2.0.57`, compatibles con las
versiones pandas/pyarrow del runtime-lock. El directorio de artefactos es
local y separado del archivo SQLite:

- Tracking/Registry: `data/mlflow/tracking.db`
- Artefactos MLflow: `data/mlflow/artifacts/`
- Reportes de M7: `data/modeling/training/reports/`

Los valores se pueden cambiar en la sección `training` de `configs/config.yaml`.
No se inicializa un servidor, no se usa un tracking URI remoto y no se usa AWS.
El experimento existente debe conservar su `artifact_path`; si se cambia, se
debe configurar otra base SQLite o restaurar la ruta anterior.

## Ejecución

Instalar las dependencias del proyecto (MLflow está fijado en `pyproject.toml`):

```powershell
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -r requirements/dev-lock.txt
python -m pip install -e ".[dev,training]" --no-deps
```

Con un reporte M6 ya producido:

```powershell
f1-pipeline --stage m7 --m6-report data/modeling/reports/modeling_report_<timestamp>_<run-id>.json
```

`--m6-report` también se puede establecer como `training.m6_report_path`. Para
examinar tracking y modelos localmente:

```powershell
mlflow ui --backend-store-uri sqlite:///data/mlflow/tracking.db
```

Recuperación para inferencia:

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
también están en el reporte M7 y en los tags del run.

## Archivos de esta entrega

- `src/f1_race_intelligence/modeling/training.py`: verificación M6, carga de
  matrices M6, baselines, métricas, registro, comprobación de recuperación y
  reporte M7.
- `src/f1_race_intelligence/config/settings.py` y `configs/config.yaml`:
  configuración local M7, aditiva a las secciones existentes.
- `src/f1_race_intelligence/cli.py`: stage `m7` y opción `--m6-report`.
- `pyproject.toml`: extra opcional `training` con MLflow y SQLAlchemy fijados.
- `tests/unit/test_training.py`: métricas con ceros/nulos y flujo completo con
  fixtures sintéticos: reproducibilidad, tracking, registro y carga.
- `README.md`: estado y enlace al alcance M7.

No se cambian splits, contratos, features ni guardas M1–M6.

## Validación y estado

Las pruebas usan datasets sintéticos con forma M5 y artefactos M6 generados en
directorios temporales. Las métricas resultantes son solo comprobaciones
técnicas: **no son resultados académicos ni describen carreras reales**.
Ninguna prueba consulta OpenF1, descarga datos o ejecuta entrenamiento sobre
datos reales.

Pendiente: ejecutar M6/M7 sobre los datos reales únicamente con autorización,
revisar suficiencia de carreras y discutir los resultados con el equipo.
LSTM/GRU y métodos estadísticos avanzados quedan fuera de alcance.

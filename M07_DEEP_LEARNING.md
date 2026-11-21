# M7 — Deep Learning (PyTorch Tabular MLP)

## Propósito y alcance académico

Esta extensión incorpora un modelo de **Deep Learning tabular** a la etapa M7 —un
perceptrón multicapa (*Multi-Layer Perceptron*, `TabularMLP`) implementado en
**PyTorch CPU-only**— entrenado directamente sobre la matriz numérica codificada
que produce M6 (`preprocessing.output_feature_names`).

El objetivo académico y de ingeniería es habilitar un modelo no lineal real
(entrenamiento por mini-batches, pérdida Huber o MSE, regularización por dropout
y *weight decay*, *early stopping* sobre `validation` y registro de curvas por
época en MLflow) manteniendo **intactos** todos los invariantes de M1–M6, la
compatibilidad hacia atrás del baseline lineal (`linear_regression`) y el
contrato extremo a extremo M6 → M7 → MLflow → publicación inmutable → M8.

### Por qué un MLP tabular y por qué LSTM/GRU queda fuera de alcance

La unidad de análisis del proyecto es **Piloto × Carrera × Vuelta** con el
objetivo de predecir `next_lap_time`. Dos decisiones de diseño explican por qué
un MLP tabular es la arquitectura adecuada en esta etapa y por qué las redes
recurrentes (LSTM/GRU) quedan deliberadamente fuera de alcance:

1. **M5 ya captura la dinámica temporal del piloto:** el dataset de features
   incluye lags y agregados móviles *trailing* calculados estrictamente dentro de
   `(session_key, driver_number)` hasta la vuelta `t` (`lap_time_prev_1`,
   `lap_time_delta_1`, `lap_time_roll_{mean,std,min}_{3,5}`,
   `lap_time_vs_roll_mean_5` y `lap_time_expanding_mean`), junto con edad y
   compuesto de neumático, ritmo de carrera y condiciones de pista.
2. **Conservación del contrato fila-a-fila en M6 y M8:** M6 produce matrices
   tabulares independientes por split (`features.parquet`), un único
   `preprocessor.joblib` y un `feature_schema.json`; M8 recibe registros
   independientes por petición (`POST /predict`) sin estado secuencial entre
   vueltas. Introducir LSTM/GRU exigiría redefinir el empaquetado de secuencias
   en M6 (ventanas de longitud variable, *padding* en stints cortos o abandonos)
   y cambiar el contrato de inferencia en línea de M8, rompiendo el aislamiento
   de etapas sin evidencia previa de que un modelo secuencial supere al tabular
   con los datos reales pendientes de autorización.

## Arquitectura del modelo e hiperparámetros

La implementación vive en
[src/f1_race_intelligence/modeling/dl.py](src/f1_race_intelligence/modeling/dl.py),
con configuración tipada en `MLPSettings`
([src/f1_race_intelligence/config/settings.py](src/f1_race_intelligence/config/settings.py))
y valores declarativos bajo `training.mlp` en
[configs/config.yaml](configs/config.yaml).

### Topología de la red (`TabularMLP`)

Dada la dimensión de entrada `D = len(preprocessing.output_feature_names)`:

```text
Entrada (D features codificadas de M6, float32)
  └─ Por cada dim en hidden_dims (por defecto [64, 32]):
       nn.Linear(prev_dim, dim) → nn.ReLU() → nn.Dropout(dropout)  [si dropout > 0]
  └─ Salida escalar:
       nn.Linear(last_hidden_dim, 1)  →  predicción escalada → inversa a segundos
```

### Hiperparámetros y valores por defecto

| Parámetro (`training.mlp.*` / `MLPConfig`) | Defecto | Restricción | Propósito |
|---|---|---|---|
| `hidden_dims` | `[64, 32]` | Lista no vacía de enteros `> 0` | Ancho de cada capa oculta totalmente conectada |
| `dropout` | `0.1` | `0.0 <= dropout < 1.0` | Regularización por dropout tras cada `ReLU` (omitida si es `0.0`) |
| `loss` | `"huber"` | `"huber"` \| `"mse"` | `"huber"` usa `nn.SmoothL1Loss(beta=1.0)`, robusta frente a outliers de boxes y safety car; `"mse"` usa `nn.MSELoss()` |
| `learning_rate` | `0.001` | Finito `> 0` | Tasa de aprendizaje de `torch.optim.Adam` |
| `weight_decay` | `0.0001` | Finito `>= 0` | Regularización L2 en `torch.optim.Adam` |
| `batch_size` | `64` | Entero `>= 1` | Tamaño de mini-batch en entrenamiento |
| `max_epochs` | `100` | Entero `>= 1` | Límite superior de épocas |
| `early_stopping_patience` | `10` | Entero `>= 1` | Épocas consecutivas sin mejora antes de detener el bucle |
| `min_delta` | `0.0001` | Finito `>= 0` | Mejora mínima exigida para considerar un nuevo mejor checkpoint |
| `scale_target` | `true` | Booleano | Estandariza `y` usando únicamente targets finitos de `train` |
| `registered_model_name` | `null` | `null` o cadena no vacía | Si es `null`, usa `"f1-next-lap-mlp"` (salvo override de `training.registered_model_name` o `--registered-model-name`) |
| `random_seed` (`modeling.random_seed`) | `42` | Entero | Semilla determinista compartida con M6/M7 |

### Entrenamiento, early stopping y reproducibilidad

- **Optimizador y mini-batches:** `torch.optim.Adam` con `lr=learning_rate` y
  `weight_decay=weight_decay`, iterando sobre un `DataLoader` en CPU con
  `shuffle=True`, `num_workers=0` y un `torch.Generator().manual_seed(seed)`.
- **Early stopping sobre `validation`:** al final de cada época se evalúa la
  pérdida media sobre las filas de `validation` con target finito en modo
  `model.eval()` y `torch.no_grad()` (por bloques de 8192 filas). Si
  `val_loss < best_val_loss - min_delta`, se actualiza `best_epoch`,
  `best_val_loss` y una copia en CPU de `state_dict`, reiniciando la paciencia.
  Al agotar `early_stopping_patience` (o completar `max_epochs`), se restauran
  los pesos del mejor checkpoint. Si `validation` no contuviera ningún target
  finito, se monitorea `train_loss` y `best_val_loss` se informa como `null`.
  Una pérdida no finita detiene la ejecución con `RuntimeError`.
- **Reproducibilidad determinista:** antes de construir la red y el cargador de
  datos se fijan `random.seed(seed)`, `np.random.seed(seed)`,
  `torch.manual_seed(seed)` y `torch.use_deterministic_algorithms(True, warn_only=True)`
  en `device = torch.device("cpu")`, restaurando el estado global de
  determinismo al terminar en un bloque `finally`. Con la misma semilla, datos,
  dependencias bloqueadas, misma plataforma, versión de PyTorch y configuración
  de hilos, los pesos, el historial y las predicciones son reproducibles bit a
  bit (PyTorch no garantiza identidad bit a bit entre plataformas distintas).

## Anti-leakage y contrato M7 → MLflow → Publicación → M8

### Guardas anti-leakage

M7 Deep Learning hereda todas las verificaciones de M6 y M7: exige backend
`local`, verifica el reporte M6 y sus ocho artefactos con `verify_report`,
acepta solo veredictos `VALID` o `VALID_WITH_WARNINGS` sin `aborted` ni
`blocking_failures`, valida `target == "next_lap_time"`, nunca reajusta
`preprocessor.joblib` ni altera los splits y nunca pasa el target ni las
columnas `key_*` como features.

Sobre el manejo específico de Deep Learning:

- **Estandarización del target únicamente con `train`:** con `scale_target=True`,
  `y_mean` y `y_std` se calculan exclusivamente sobre los targets finitos del
  split `train` (las filas sin target finito se descartan antes de ajustar). Si
  la desviación estándar es menor que `1e-8` (`MIN_TARGET_STD`), se fija
  `y_std = 1.0` para evitar divisiones por cero ante targets constantes.
  `validation` se transforma con el `y_mean` y `y_std` de `train` únicamente
  para medir `val_loss` y decidir el *early stopping*; `test` jamás participa en
  entrenamiento, escalado ni selección de época.
- **Transformación inversa embebida en el modelo (regla 2 del contrato M7 → M8):**
  tanto `FittedTabularMLP.predict` como `TabularMLPPyfuncModel.predict` aplican
  `pred = scaled.astype(np.float64) * y_std + y_mean` antes de devolver el
  resultado. Cumpliendo la regla 2 de
  [README.md §19](README.md#contrato-de-integración-con-m7), el modelo registrado
  siempre devuelve `next_lap_time` en segundos (escala original de
  `lap_duration`).

### Serialización limpia y compatibilidad con publicación y M8

El modelo se serializa y registra en MLflow cumpliendo estrictamente el contrato
validado por
[src/f1_race_intelligence/modeling/model_package.py](src/f1_race_intelligence/modeling/model_package.py)
y [src/f1_race_intelligence/serving/service.py](src/f1_race_intelligence/serving/service.py):

- **Checkpoint autocontenido y seguro (`mlp_checkpoint.pt`):**
  `save_mlp_checkpoint` guarda únicamente tensores CPU y valores primitivos de
  Python (`format_version`, `state_dict`, `feature_names`, `hidden_dims`,
  `dropout`, `y_mean`, `y_std`, `scale_target`). `load_mlp_checkpoint` lo lee
  siempre con `torch.load(path, map_location="cpu", weights_only=True)` y valida
  la versión de formato y claves obligatorias.
- **Wrapper `mlflow.pyfunc.PythonModel`:** `TabularMLPPyfuncModel` se registra
  con `mlflow.pyfunc.log_model` sin estado mutable en el pickle; en
  `load_context` reconstruye `TabularMLP` desde el artefacto `checkpoint` en
  modo `.eval()`.
- **Tipos de entrada/salida y firma nombrada:** la firma se infiere sobre un
  ejemplo ordenado exactamente por `output_feature_names`
  (`infer_signature(example, fitted.predict(example))`), de modo que
  `metadata.get_input_schema().input_names()` coincide exactamente en nombre y
  orden con `feature_schema.json["encoded_feature_names"]`. En inferencia,
  `TabularMLPPyfuncModel.predict` recibe el `pd.DataFrame` `float64` producido
  por M8 (`PredictionService.encode`), selecciona las columnas en orden, castea
  internamente a `float32` para PyTorch y devuelve un `np.ndarray` 1D `float64`
  finito en segundos.
- **Requisitos de empaquetado inmutable:** el directorio `model/` registrado
  (con `MLmodel` y `artifacts/mlp_checkpoint.pt`) no contiene symlinks, todos
  sus nombres de archivo cumplen `[A-Za-z0-9][A-Za-z0-9._-]*` y
  `metadata.run_id` coincide con el `run_id` del run M7. Al cerrar el run, M7
  recarga `models:/<nombre>/<versión>` mediante `mlflow.pyfunc.load_model` y
  comprueba reproducción numérica (`rtol=1e-5, atol=1e-5`).
- **Publicación y serving sin adaptadores especiales:** `publish_model` y
  `PredictionService.load_published_package` funcionan exactamente igual para un
  modelo `mlp` que para `linear_regression` (ver
  [M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md)), sin volver a ajustar
  `ColumnTransformer`.

### Comparación justa y trazabilidad en MLflow

Cada corrida de M7 con `--model-type mlp` ajusta en el mismo run —sobre el mismo
split `train` del mismo reporte M6 y con idéntico `input_fingerprint`— los dos
baselines de referencia (`DummyRegressor(strategy="mean")` y `LinearRegression`)
junto con `TabularMLP`. Así quedan registrados bajo el mismo run y experimento:

- **Tags:** `project`, `stage=m7-training`, `model_type=mlp`, `data_version`,
  `input_fingerprint`, `m6_run_id`, `m6_source_commit`, `m7_source_commit`.
- **Parámetros:** `model_type`, `estimator`
  (`f1_race_intelligence.modeling.dl.TabularMLP`), `reference_baseline`,
  `random_seed`, `m6_run_id`, `m6_pipeline_version`, `input_fingerprint`,
  `source_commit`, además de `mlp_hidden_dims`, `mlp_dropout`,
  `mlp_learning_rate`, `mlp_weight_decay`, `mlp_batch_size`, `mlp_max_epochs`,
  `mlp_early_stopping_patience`, `mlp_min_delta`, `mlp_loss`,
  `mlp_scale_target`, `mlp_total_params`, `mlp_best_epoch` y
  `mlp_epochs_trained`.
- **Métricas por época:** `train.mlp.loss` y `validation.mlp.loss` (con
  `step=epoch`), además del artefacto `training_history.json`.
- **Métricas finales por split (`train`, `validation`, `test`):** `mae`, `rmse`,
  `mape`, `r2`, `rows`, `valid_rows`, `excluded_rows` y `zero_target_rows` para
  los tres modelos (`mlp`, `linear_regression` y `reference_mean`).
- **Artefactos obligatorios de trazabilidad:** `training_config.json`,
  `data_provenance.json`, `metrics.json`, `training_report.json` (con la sección
  adicional `mlp_training`), `training_history.json`, `m6/<modeling_report>`,
  `m6/preprocessor.joblib`, `m6/feature_schema.json` y `model/`.

## Aislamiento de dependencias, portabilidad a AWS y por qué no SageMaker

### Aislamiento del runtime base (M1–M6)

- [requirements/runtime-lock.txt](requirements/runtime-lock.txt) y
  [Dockerfile](Dockerfile) (imagen batch M2–M6) **no se modifican** y no
  incluyen `mlflow` ni `torch`.
- Todos los imports de `torch` en `modeling/dl.py` son perezosos (*lazy*
  mediante `_require_torch()`); la clase base `mlflow.pyfunc.PythonModel` usa un
  *stand-in* cuando `mlflow` no está instalado. Invocar M7 MLP sin PyTorch falla
  explícitamente con un `RuntimeError` que indica cómo instalar el extra.
- En CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)), el job
  `base-runtime` instala únicamente `runtime-lock.txt` y `dev-lock.txt`,
  comprueba explícitamente que ni `mlflow` ni `torch` estén presentes y ejecuta
  las pruebas M1–M6 y de dependencias opcionales; el job `tests` instala además
  `training-lock.txt` y `dl-lock.txt` en Ubuntu y Windows.

### Por qué CPU local hoy y por qué no Amazon SageMaker

1. **Tamaño real del dataset académico (temporadas 2023–2025):** tres temporadas
   de carreras de F1 (~23–24 carreras/año × ~20 pilotos × ~55–65 vueltas)
   equivalen a un orden de **~70 000 – 90 000 filas** con ~60–100 columnas
   codificadas tras M6. Un MLP compacto (`[64, 32]`, unos pocos miles de
   parámetros) con mini-batches de 64 y *early stopping* tiene un tiempo
   estimado de entrenamiento del orden de **decenas de segundos en CPU local**
   (estimación de orden de magnitud, no una métrica medida sobre datos reales).
2. **Sin sobre-ingeniería ni coste cloud innecesario:** montar entrenamiento
   administrado en Amazon SageMaker (o instancias GPU) introduciría costes
   recurrentes, permisos IAM dedicados, infraestructura adicional y latencia de
   aprovisionamiento que no aportan ventaja medible sobre este volumen tabular ni
   son necesarios mientras el entrenamiento real y el despliegue AWS sigan
   pendientes.
3. **Portabilidad directa hacia AWS si crece el alcance:** el extra opcional
   `dl` ([pyproject.toml](pyproject.toml)) fija `torch==2.12.1` y
   [requirements/dl-lock.txt](requirements/dl-lock.txt) bloquea los wheels
   **CPU-only** (`torch==2.12.1+cpu` para Linux x86_64/arm64 y Windows mediante
   `--extra-index-url https://download.pytorch.org/whl/cpu`; `torch==2.12.1` en
   macOS Apple Silicon ≥ 14 mediante el wheel `macosx_14_0_arm64` —en Mac Intel
   o macOS < 14 no existe wheel precompilado de `torch 2.12.1`), evitando
   dependencias CUDA pesadas y manteniendo la compatibilidad exacta con
   `runtime-lock.txt` y `training-lock.txt`. Como el checkpoint carga siempre en
   CPU (`map_location="cpu"`) y la CLI es autocontenida, el mismo stage puede
   llevarse más adelante a una tarea batch en **ECS Fargate CPU** (en una imagen
   derivada que sume `training-lock.txt` y `dl-lock.txt`) o a un **SageMaker
   PyTorch Estimator** sin reescribir el modelo, la publicación ni M8.

## Instalación y comandos de ejecución

### Instalación de dependencias bloqueadas

Para desarrollo, entrenamiento lineal/DL, pruebas y serving local:

```bash
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -r requirements/dev-lock.txt
python -m pip install -r requirements/dl-lock.txt
python -m pip install -e ".[dev,training,serve,dl]" --no-deps
```

Si solo se necesita ejecutar entrenamiento M7 (lineal y MLP) sin las herramientas
de desarrollo ni FastAPI:

```bash
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -r requirements/dl-lock.txt
python -m pip install -e ".[training,dl]" --no-deps
```

> `requirements/dl-lock.txt` selecciona automáticamente `torch==2.12.1+cpu` en
> Linux y Windows, y `torch==2.12.1` en **macOS Apple Silicon ≥ 14** (wheel
> `macosx_14_0_arm64`; en Mac Intel o macOS < 14 no existe wheel precompilado
> de `torch 2.12.1`).

### Entrenamiento M7 en modo MLP

Con un reporte M6 verificado previamente producido:

```bash
f1-pipeline --stage m7 --model-type mlp \
    --m6-report data/modeling/reports/modeling_report_<timestamp>_<run-id>.json
```

También puede invocarse con `python -m f1_race_intelligence` o fijar
`training.model_type: "mlp"` y los hiperparámetros `training.mlp.*` en
[configs/config.yaml](configs/config.yaml). Para usar un nombre de modelo
registrado distinto del valor por defecto (`f1-next-lap-mlp`):

```bash
f1-pipeline --stage m7 --model-type mlp \
    --registered-model-name f1-next-lap-mlp-exp \
    --m6-report data/modeling/reports/modeling_report_<timestamp>_<run-id>.json
```

### Inspección local en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///data/mlflow/tracking.db
```

En la interfaz se pueden comparar directamente las métricas de `mlp`,
`linear_regression` y `reference_mean`, inspeccionar las curvas
`train.mlp.loss` y `validation.mlp.loss` por época, y consultar el modelo
registrado bajo `f1-next-lap-mlp`.

### Recuperación, publicación inmutable y serving M8

Carga directa desde el registro local de MLflow (en segundos):

```python
import mlflow
import mlflow.pyfunc

mlflow.set_tracking_uri("sqlite:///data/mlflow/tracking.db")
model = mlflow.pyfunc.load_model("models:/f1-next-lap-mlp/<version>")
predictions = model.predict(m6_encoded_features[encoded_feature_names])
```

Publicación inmutable y carga verificada desde M8 (ver detalle completo en
[M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md)):

```python
import mlflow
from f1_race_intelligence.modeling.model_package import publish_model
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import PredictionService
from f1_race_intelligence.storage.backends.local import LocalStorageBackend

mlflow.set_tracking_uri("sqlite:///data/mlflow/tracking.db")
mlflow.set_registry_uri("sqlite:///data/mlflow/tracking.db")

backend = LocalStorageBackend("model-store")
package = publish_model("models:/f1-next-lap-mlp/<version>", backend)
service = PredictionService.load_published_package(
    backend=backend,
    prefix=package.prefix,
    loader=MlflowPyfuncLoader(),
)
```

Para levantar el servidor HTTP local de M8 contra un run M7 MLP registrado (con
el extra `dl` instalado):

```bash
export MLFLOW_TRACKING_URI=sqlite:///data/mlflow/tracking.db
export F1_SERVING_PREPROCESSING_URI=runs:/<run_id de M7>/m6
export F1_SERVING_MODEL_URI=models:/f1-next-lap-mlp/<version>
uvicorn f1_race_intelligence.serving.app:app --host 127.0.0.1 --port 8000
```

## Validación, pruebas y honestidad académica

Las suites que validan M7 Deep Learning se ejecutan localmente sin acceso a red:

```bash
# Pruebas unitarias del MLP: determinismo, Huber/MSE, anti-leakage del target,
# early stopping, import perezoso sin torch/mlflow, checkpoint weights_only=True y pyfunc
python -m pytest tests/unit/test_dl.py

# Integración de TrainingRunner (lineal y MLP), publicación inmutable y serving M8
python -m pytest tests/unit/test_training.py tests/unit/test_dl_m7_m8_integration.py tests/unit/test_m7_m8_integration.py

# Suite completa no integrada (excluye tests/integration contra OpenF1 real)
python -m pytest -m "not integration"

# Demostración integral local M1–M8 (flujo lineal y flujo Deep Learning)
python -m pytest scripts/demo_local_mlops.py -s -p no:cacheprovider -p no:warnings
```

**Declaración de honestidad académica:**

- Todas las pruebas unitarias, pruebas de integración M7–M8 y la demostración en
  [DEMO_LOCAL.md](DEMO_LOCAL.md) utilizan exclusivamente **datos sintéticos** y
  un cliente OpenF1 simulado.
- Las pérdidas por época y las métricas de regresión obtenidas sobre dichos
  fixtures verifican únicamente la corrección técnica del pipeline (flujo de
  datos, guardas anti-leakage, trazabilidad, serialización y serving) y **no son
  resultados académicos reales** ni representan desempeño predictivo sobre
  carreras reales de Fórmula 1.
- Ninguna prueba ni comando aquí validado consulta la API real de OpenF1 ni
  despliega recursos en AWS.
- Los resultados académicos reales —incluyendo la evaluación empírica de
  `TabularMLP` frente a `LinearRegression` y `DummyRegressor(strategy="mean")`
  sobre el histórico 2023–2025— quedan pendientes de ejecutar M2–M6 y M7 sobre
  datos reales de OpenF1 una vez se cuente con autorización.

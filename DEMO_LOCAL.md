# Demostración local M1–M8 (datos sintéticos, sin red)

Esta guía reproduce, de extremo a extremo y **sin llamar a OpenF1 ni a AWS**,
la cadena MLOps del proyecto: extracción por temporadas (M2), validación (M3),
consolidación (M4), features (M5), dataset de modelado (M6), entrenamiento y
registro (M7 —tanto regresión lineal como Deep Learning MLP en PyTorch—),
publicación del modelo y serving (M8).

La demostración es un único módulo pytest,
[scripts/demo_local_mlops.py](scripts/demo_local_mlops.py), con dos flujos
integrales:

- **`test_integral_flow`**: recorre M1–M8 usando el modelo por defecto de M7
  (`linear_regression`).
- **`test_integral_dl_flow`**: recorre M1–M8 usando Deep Learning en M7
  (`--model-type mlp`, PyTorch CPU-only tabular MLP; ver
  [M07_DEEP_LEARNING.md](M07_DEEP_LEARNING.md)). Se ejecuta cuando está
  instalado el extra opcional `dl` (`requirements/dl-lock.txt`); si `torch` no
  está instalado, pytest lo omite automáticamente.

Ambos flujos usan el **código real del proyecto** (la CLI
`python -m f1_race_intelligence`, MLflow, PyTorch, la publicación inmutable y
la API FastAPI) y sustituyen solo la fuente de datos: un cliente OpenF1
simulado (`SeasonClient`, en `tests/unit/test_backfill.py`) que sirve tres
temporadas sintéticas (2023, 2024 y 2025) de dos carreras cada una. Ningún
informe se fabrica: cada etapa escribe los suyos.

## Qué demuestra y qué no

| Demuestra (localmente) | No demuestra |
|---|---|
| Backfill por temporadas con `F1_EXTRACTION_YEARS`; M2–M5 por sesión | Datos, volúmenes ni duraciones reales de OpenF1 |
| M6 **diferido** (código de salida 3) con el histórico incompleto y **ejecutado** con el completo | Rendimiento predictivo: las métricas (lineales y del MLP) salen de 6 carreras artificiales y **no son resultados académicos** |
| M7 por CLI (`linear_regression` y `--model-type mlp`), registro en MLflow (incluyendo curvas por época y baselines comparativos) y recuperación de la ejecución | MLflow remoto ni entrenamiento en AWS/SageMaker: aquí es SQLite local y PyTorch en CPU local |
| Publicación inmutable del paquete (lineal y MLP) y verificación de su manifiesto y hashes | Controles de S3 real (IAM, versionado, Object Lock) |
| M8 cargando el paquete publicado (lineal y MLP `pyfunc`), sin reajustar el preprocesador | Despliegue de M7/M8 en AWS (no existe) |
| La cadena de trazabilidad M6 → M7 → MLflow → paquete → M8 | La orquestación AWS (Step Functions, admisión, alarmas): solo validada estáticamente y por simulación |

Los datos sintéticos hacen que M3 emita advertencias (`PARTIAL`) en las
sesiones: es esperado y no una fuga ni un error.

## Requisitos

- Python 3.11 o superior (la imagen usa 3.12), `git` y este repositorio.
- Dependencias fijadas del proyecto, incluyendo el lock CPU-only de PyTorch
  para cubrir tanto `test_integral_flow` como `test_integral_dl_flow`:

```bash
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -r requirements/dev-lock.txt
python -m pip install -r requirements/dl-lock.txt
python -m pip install -e ".[dev,training,serve,dl]" --no-deps
```

Si no se instala `requirements/dl-lock.txt` ni el extra `dl` (es decir, solo
`pip install -e ".[dev,training,serve]" --no-deps`), `test_integral_flow` se
ejecuta con normalidad y `test_integral_dl_flow` queda omitido por falta de
`torch`.

No hacen falta credenciales ni variables de entorno: la demo aísla el entorno
(`F1_*`) y trabaja en directorios temporales.

## Ejecución

Desde la **raíz del repositorio**:

```bash
python -m pytest scripts/demo_local_mlops.py -s -p no:cacheprovider -p no:warnings
```

También se puede ejecutar de forma focalizada solo uno de los flujos:

```bash
# Solo baseline lineal (M1–M8)
python -m pytest scripts/demo_local_mlops.py -k test_integral_flow -s -p no:cacheprovider -p no:warnings

# Solo Deep Learning MLP en PyTorch (M1–M8, requiere extra dl)
python -m pytest scripts/demo_local_mlops.py -k test_integral_dl_flow -s -p no:cacheprovider -p no:warnings
```

`-s` muestra las líneas `[DEMO]` que se citan abajo. Para conservar y poder
abrir los artefactos (MLflow, paquete publicado), fija un directorio de trabajo
fuera del repositorio:

```bash
python -m pytest scripts/demo_local_mlops.py -s -p no:cacheprovider -p no:warnings --basetemp "<carpeta>/f1-demo"
```

Con `--basetemp`, pytest vacía y recrea esa carpeta. Los artefactos quedan en
`<carpeta>/f1-demo/test_integral_flow0/` (baseline lineal) y
`<carpeta>/f1-demo/test_integral_dl_flow0/` (Deep Learning MLP), cada uno con
su propia estructura (`mlflow/tracking.db`, `model-store/published-models/…`,
`modeling/`, `features/`, `raw/`…). Se puede inspeccionar cualquiera de los dos
con `mlflow ui --backend-store-uri sqlite:///<ruta>/mlflow/tracking.db`.

Con `requirements/dl-lock.txt` y `.[dev,training,serve,dl]` instalados, el
módulo ejecuta ambos flujos (`2 passed`); sin `torch`, ejecuta únicamente el
flujo lineal (`1 passed, 1 skipped`). Si algo no concuerda, el módulo falla con
la aserción correspondiente.

## Secuencia y criterios de éxito

Cada fase lanza el **plan** con `F1_EXTRACTION_YEARS=<año>` (así se acota una
fase, como hace la state machine), ejecuta M2→M3→M4→M5 por cada sesión
planificada con la CLI real y llama a
`python -m f1_race_intelligence --stage m6 --require-complete-history`.

| Paso | Comando real que invoca | Criterio de éxito |
|---|---|---|
| Fase 2023 | `--stage plan`, luego `--stage m2/m3/m4/m5 --year … --meeting-key … --session-key …` | 2 sesiones planificadas; M2–M5 con código `0, 0, 0, 0` |
| M6 con histórico incompleto | `--stage m6 --require-complete-history` | **código 3**; 0 informes de M6; `[DEMO] phase 2023: … m6 exit=3 … m6 reports so far=0` |
| Fase 2024 | igual | M2–M5 en `0`; M6 de nuevo con **código 3** y sin informes |
| Fase 2025 | igual | M2–M5 en `0`; M6 con **código 0**; 1 informe |
| Histórico | `history_completeness` | `{'2023': ('complete', 2), '2024': ('complete', 2), '2025': ('complete', 2)}` |
| M6 | `verify_report` | veredicto `VALID`, `sessions=6`, 8 artefactos, `verify_report ok=True` |
| M7 (lineal / DL) | `--stage m7 --m6-report <informe>` (y `--model-type mlp` en `test_integral_dl_flow`) | código `0`; versión 1 registrada en MLflow |
| MLflow | `MlflowClient` | versión 1 en estado `READY`, run `FINISHED`; `list_artifacts` devuelve `data_provenance.json`, `m6`, `metrics.json`, `training_config.json` y `training_report.json` (más `training_history.json` y curvas por época en DL); en MLflow 3 el modelo queda como `LoggedModel` accesible vía `runs:/<id>/model` y `models:/<nombre>/<versión>` aunque `list_artifacts` no liste `model` |
| Publicación | `publish_model(...)` y `read_model_package(...)` | paquete verificado con manifiesto, hashes y marcador `PUBLISHED`; republicar es idempotente |
| M8 | `PredictionService.load_published_package` + `TestClient` | `/live`, `/health`, `/model-info` y `/predict` con **200**; `/predict` devuelve predicciones finitas de `next_lap_time` en segundos |

### Flujo Deep Learning (`test_integral_dl_flow`)

El flujo `test_integral_dl_flow` reproduce exactamente las fases M2–M6
anteriores (incluyendo el diferimiento `exit=3 → 3 → 0` de M6) y ejercita de
extremo a extremo la variante Deep Learning sobre el mismo contrato MLOps:

- Ejecuta M7 mediante la CLI real con
  `--stage m7 --model-type mlp --registered-model-name integral-dl-demo-model --m6-report <informe>`.
- Verifica en MLflow local el registro de `models:/integral-dl-demo-model/1`
  (`mlflow.pyfunc` con checkpoint PyTorch `mlp_checkpoint.pt` cargado con
  `weights_only=True` en CPU), las curvas de pérdida por época
  (`train.mlp.loss`, `validation.mlp.loss`, artefacto `training_history.json`)
  y las métricas comparativas de `mlp`, `linear_regression` y `reference_mean`
  sobre el mismo reporte M6 (comprobando además que `y_mean` coincide con la
  media de los targets finitos de `train`).
- Publica el modelo MLP como paquete inmutable tanto en almacenamiento local
  como en un S3 simulado en memoria (`FakeS3Client`), verificando igualdad
  exacta de manifiesto y artefactos, ausencia de symlinks, marcador `PUBLISHED`
  y republicación idempotente.
- Lo sirve en M8 vía `PredictionService.load_published_package` con
  `ColumnTransformer.fit` y `fit_transform` bloqueados, comprobando respuestas
  `200` en `/live`, `/health`, `/model-info` y `/predict` (predicciones finitas
  de `next_lap_time` en segundos gracias a la transformación inversa embebida
  en el modelo), paridad numérica contra la carga por `runs:/<m7_run_id>/m6` y
  la identidad completa de la cadena M6 → M7 → MLflow → paquete → M8.

### Cómo ver el diferimiento de M6

En las líneas de las fases 2023 y 2024 aparece `m6 exit=3` y
`m6 reports so far=0`: M6 no escribió ningún artefacto de modelado porque la
evidencia almacenada no demuestra las tres temporadas completas. Es una salida
controlada (no un fallo). En la fase 2025 la misma orden devuelve `0` y crea el
informe. La evidencia que se evalúa (catálogo, manifiesto M2, hashes de raw,
informes y salidas de M3–M5 por sesión) está descrita en
[infra/README.md](infra/README.md#backfill-histórico-por-fases-r2).

### Cómo ver la trazabilidad

Las líneas `[DEMO] chain:` muestran los mismos identificadores en cada eslabón
y ambos flujos lo comprueban con aserciones:

- el `run_id` de M6 aparece en el informe de M7, en el manifiesto del paquete y
  en la procedencia que expone `/model-info`;
- el `input_fingerprint` (`sha256:c2a462e7…7600140` en esta ejecución) es el
  mismo en M6, M7, el manifiesto y M8;
- el `run_id` de M7 coincide con el de la versión de MLflow, el del manifiesto y
  el del modelo que sirve M8;
- la URI exacta registrada (`models:/<nombre>/1`, por ejemplo
  `models:/integral-demo-model/1` en el flujo lineal) coincide en el paquete y
  en M8;
- M8 carga el `preprocessor.joblib` **del paquete** (su hash en el manifiesto es
  el del run de MLflow) y `ColumnTransformer.fit` está bloqueado durante M8:
  no se reajusta nada.

## Resultados esperados

Ejemplo de valores observados en el flujo baseline `test_integral_flow` (los
identificadores `run_id` cambian en cada ejecución; el `input_fingerprint` y
las predicciones dependen solo de los datos sintéticos, la semilla y las
versiones de las dependencias):

```
[DEMO] phase 2023: … m6 exit=3; m6 reports so far=0
[DEMO] phase 2024: … m6 exit=3; m6 reports so far=0
[DEMO] phase 2025: … m6 exit=0; m6 reports so far=1
[DEMO] M6: … verdict=VALID sessions=6 … verify_report ok=True artifacts=8
[DEMO] M7 exit: 0
[DEMO] MLflow registry: [('integral-demo-model', 1, 'READY', '<run_id de M7>')]
[DEMO] GET /live: (200, {'status': 'ok'})   /health y /model-info también 200
[DEMO] POST /predict: (200, [108.212, 109.912, 113.802], 'next_lap_time')
```

En `test_integral_dl_flow` la secuencia de salida es análoga, con M7 ejecutado
en modo `--model-type mlp`, versión registrada en MLflow `READY` con
`training_history.json` en los artefactos del run y el modelo (`pyfunc` +
`mlp_checkpoint.pt`) como `LoggedModel` accesible vía `runs:/<id>/model` y
`models:/integral-dl-demo-model/1`, paquete inmutable verificado y
`POST /predict` devolviendo `200` con predicciones finitas de `next_lap_time`
en segundos.

## Verificaciones relacionadas

La demostración no sustituye a las pruebas; las complementa:

```bash
python -m pytest -m "not integration"      # suite completa no integrada
python -m pytest tests/unit/test_dl.py     # MLP PyTorch: determinismo, anti-leakage, pyfunc
python -m pytest tests/unit/test_training.py tests/unit/test_dl_m7_m8_integration.py tests/unit/test_m7_m8_integration.py
python -m pytest tests/unit/test_backfill.py tests/unit/test_incremental_e2e.py
python -m pytest tests/infra               # plantilla, ASL y Lambda de admisión
```

En la validación integral previa de `master` (antes de incorporar Deep Learning
en M7), la suite no integrada dio **979 passed y 2 deselected** (las dos
pruebas de `tests/integration` llaman a OpenF1 real y están excluidas por
defecto). Con la extensión DL (`tests/unit/test_dl.py`,
`tests/unit/test_dl_m7_m8_integration.py` y los casos DL añadidos en
`test_training.py`, `test_cli.py` y `test_settings.py`), ejecuta
`python -m pytest -m "not integration"` para verificar el total actualizado en
tu entorno. `scripts/` no se recoge en `pytest` sin una ruta explícita, por lo
que la demostración no cuenta en esa cifra.

## Limitaciones de la demostración

- Es **offline y sintética**: los números de ambos modelos (regresión lineal y
  MLP tabular PyTorch sobre 6 carreras artificiales) solo prueban que la cadena
  funciona, no su capacidad predictiva ni constituyen resultados académicos.
- MLflow usa **SQLite local** y PyTorch corre en **CPU local** (sin GPU ni
  SageMaker); el entrenamiento rechaza explícitamente S3.
- La publicación se ejecuta sobre el backend local; el backend S3 está probado
  únicamente con un S3 simulado.
- El servicio M8 se ejercita con el cliente de pruebas de FastAPI (en proceso),
  no como servidor `uvicorn` desplegado.
- La orquestación AWS (Step Functions, Lambda de admisión, EventBridge, alarmas,
  dashboard) **no forma parte de la demo**: está validada estáticamente
  (`sam validate --lint`, pruebas de plantilla y ASL) y por simulación, no en
  AWS real.
- Estado de las auditorías independientes (Antigravity): R1, R1.1 y R1.2
  *APPROVED WITH MINOR FINDINGS* (hallazgo menor corregido); R2 (backfill y
  protección de M6) y R3 (robustez ante el resultado de ECS) *APPROVED*. Eso no
  sustituye la validación en AWS real, que sigue pendiente.

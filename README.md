# F1 Race Intelligence

**F1 Race Intelligence** es un proyecto académico (Maestría en Ciencia de Datos y
Analítica) orientado a analizar y, en etapas futuras, predecir el desempeño de
pilotos de Fórmula 1 a partir de datos históricos de la API pública
[OpenF1](https://openf1.org/) (documentación: https://openf1.org/docs/).

La unidad de análisis objetivo es **Piloto × Carrera × Vuelta**, con la meta
de predecir `next_lap_time` (tiempo de la siguiente vuelta). M7 incorpora un
pipeline reproducible de regresión lineal y evaluación sobre los splits M6; el
entrenamiento con datos reales sigue pendiente de autorización. La cadena
completa M1–M8 está demostrada **localmente con datos sintéticos**
([DEMO_LOCAL.md](DEMO_LOCAL.md)); ver la
[visión general MLOps](#visión-general-mlops-m1m8) y su índice de documentación.

## Visión general MLOps (M1–M8)

**Propósito.** Construir de forma reproducible y trazable el camino completo
desde los datos públicos de OpenF1 hasta un modelo servido, para predecir
`next_lap_time` por Piloto × Carrera × Vuelta. El énfasis académico está en la
ingeniería: cada etapa deja evidencia verificable (manifiestos, informes con
SHA-256, `input_fingerprint`), las guardas anti-leakage no se relajan y cada
paso puede repetirse sin duplicar ni corromper resultados.

**Estado, con exactitud.**

- Implementado y probado **localmente** con datos sintéticos y un cliente
  OpenF1 simulado: M1–M8, la planificación incremental, el backfill por
  temporadas, la protección de M6, la publicación del modelo y el serving.
  La última validación integral: **979 pruebas pasadas y 2 excluidas** (las
  dos de `tests/integration`, que llaman a OpenF1 real) y la demostración
  [DEMO_LOCAL.md](DEMO_LOCAL.md).
- **No** se ha entrenado con datos reales de OpenF1. Las métricas de la
  demostración salen de seis carreras artificiales y no son resultados
  académicos de rendimiento.
- La infraestructura AWS (SAM/Step Functions/ECS/Lambda/alarmas) está definida
  en código y validada estáticamente (`sam validate --lint`) y por simulación;
  **no está desplegada ni se ha validado en AWS real**
  ([infra/aws_status.json](infra/aws_status.json)). M7 y M8 no existen en AWS:
  MLflow usa SQLite local y el serving es un prototipo local.
- Auditorías independientes (Antigravity): la planificación incremental y la
  admisión (R1, R1.1 y R1.2) obtuvieron *APPROVED WITH MINOR FINDINGS* (su único
  hallazgo menor, `pythonpath = ["src"]` en `pyproject.toml`, está corregido);
  el backfill por temporadas y la protección de M6 (R2) y la robustez ante el
  resultado de ECS (R3) obtuvieron *APPROVED*. La validación en AWS real sigue
  pendiente.

### Componentes y responsabilidades

| Etapa | Responsabilidad | Entrada → salida | Detalle |
|---|---|---|---|
| M1 | Cliente `F1Client`, configuración y almacenamiento | OpenF1 → respuestas raw | §2, §6–§8 |
| M2 | Extracción histórica idempotente por sesión | OpenF1 → `raw/` + manifiesto | §9 |
| M3 | Validación de calidad (no modifica el raw) | `raw/` + manifiesto → informe PASS/WARNING/FAIL | §10 |
| M4 | Consolidación a Piloto × Carrera × Vuelta | `raw/` + informe M3 → dataset por sesión | §11 |
| M5 | Features y target `next_lap_time`, con guardas anti-leakage | dataset M4 → features por sesión | §12 |
| M6 | Dataset de modelado: split cronológico por carrera y preprocesado ajustado solo con train | features de **todo** el histórico → splits, preprocesador y esquema con hashes | §13 |
| M7 | Entrenamiento (regresión lineal y referencia), métricas, MLflow y registro | informe M6 verificado → modelo registrado | [M07_TRAINING_MLFLOW.md](M07_TRAINING_MLFLOW.md) |
| Publicación | Paquete inmutable y verificable de una versión registrada | modelo + artefactos M6 → `published-models/…` con manifiesto y `PUBLISHED` | [M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md) |
| M8 | Serving FastAPI (`/live`, `/health`, `/model-info`, `/predict`) | paquete publicado (o URIs de MLflow) → predicciones | §19 |

```
OpenF1 ──F1Client──▶ M2 raw ──▶ M3 validación ──▶ M4 consolidado ──▶ M5 features ─┐
                       (por sesión, idempotente, con evidencia por etapa)           │
                                                                                    ▼
        M8 serving ◀── paquete publicado ◀── MLflow (registro) ◀── M7 ◀── M6 (historia completa)
```

La cadena de trazabilidad conserva el mismo `run_id` de M6, el mismo
`input_fingerprint` y el mismo `run_id` de M7 en el informe de M7, MLflow, el
manifiesto del paquete y la respuesta de `/model-info`. M8 usa el preprocesador
publicado; no reajusta nada.

### Incrementalidad, idempotencia y admisión

- **Plan incremental.** Cada sesión se omite solo si su estado
  `manifests/incremental/…` demuestra, con hashes, que M2–M5 terminaron y que
  el raw, los informes, las salidas y la configuración siguen intactos. Una
  evidencia ausente o alterada vuelve a incluirla. Un 404 legítimo de un
  endpoint solo cuenta si el manifiesto de M2 lo registra (código 404, no
  reintentable).
- **M6 idempotente.** Se reutiliza únicamente un resultado previo con veredicto
  `VALID` cuyo `input_fingerprint`, configuración, código y dependencias
  coinciden y cuyos artefactos verifican.
- **Admisión.** Una Lambda reserva en S3, con escritura condicional
  (compare-and-set), el intervalo mínimo (24 h como mínimo) entre inicios
  automáticos del pipeline. Es una espera mínima entre reservas, no una ventana
  de calendario; los inicios manuales no pasan por ella.

### Backfill por temporadas y protección de M6

El histórico se procesa **una temporada por ejecución** (`extraction_years` en
el input o `F1_EXTRACTION_YEARS` en la tarea de planificación); cada fase es
reanudable porque se apoya en el plan incremental. M6 necesita todo el
histórico para construir sus splits, así que la tarea M6 se ejecuta siempre con
`--require-complete-history`: solo corre si la evidencia almacenada prueba
completas todas las temporadas de modelado. Si no, sale con **código 3**, no
escribe artefactos y la state machine termina en
`ModelingDeferredHistoryIncomplete` (sin alerta y sin declarar completo el
backfill). Ver [infra/README.md](infra/README.md#backfill-histórico-por-fases-r2).

### Observabilidad y tratamiento de errores

Logs JSON con `component` y `pipeline_execution_id` (propagado desde Step
Functions a todas las tareas Fargate), métricas EMF de disponibilidad de
OpenF1, seis alarmas CloudWatch, un dashboard y SNS para avisos. Los resultados
de ECS se validan antes de usarse: un `ExitCode` numérico 0 es el único éxito;
otro número, la ausencia de `Containers`/`ExitCode` o un tipo incorrecto
terminan en `NotifyPipelineFailure`. Ver
[infra/README.md](infra/README.md#observabilidad-operativa).

### Índice de documentación

| Tema | Documento |
|---|---|
| Arquitectura del código y del pipeline | §2 y §3 de este README |
| M1: cliente, configuración, rate limiting | §5–§8 de este README |
| M2: extracción histórica | §9 |
| M3: validación | §10 |
| M4: consolidación | §11 |
| M5: features y target | §12 |
| M6: dataset de modelado y guardas anti-leakage | §13 |
| M7: entrenamiento y MLflow | [M07_TRAINING_MLFLOW.md](M07_TRAINING_MLFLOW.md) |
| Publicación de modelos | [M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md) |
| M8: serving | §19 de este README |
| Preparación para la nube y Docker | §14 |
| Infraestructura AWS (SAM, IAM, parámetros) | [infra/README.md](infra/README.md) |
| Planificación incremental y admisión | [infra/README.md](infra/README.md#planificación-incremental-e-idempotencia-m6-r1) |
| Backfill y protección de M6 | [infra/README.md](infra/README.md#backfill-histórico-por-fases-r2) |
| Observabilidad, alarmas y SNS | [infra/README.md](infra/README.md#observabilidad-operativa) |
| Despliegue, eliminación y riesgos | [infra/README.md](infra/README.md#desplegar-más-adelante) y [Riesgos conocidos](infra/README.md#riesgos-conocidos) |
| Demostración local reproducible | [DEMO_LOCAL.md](DEMO_LOCAL.md) |
| Estado de verificación AWS | [infra/aws_status.json](infra/aws_status.json) |

## 1. Objetivo de esta primera etapa

Esta etapa construye únicamente la base de ingeniería de datos del proyecto:

- Arquitectura de repositorio preparada para evolucionar a un pipeline MLOps.
- Un cliente Python (`F1Client`) que encapsula **toda** la comunicación HTTP
  con OpenF1: construcción de URLs, timeouts, rate limiting, retries con
  backoff exponencial, manejo de errores y logging estructurado.
- Una capa de configuración externa (YAML + variables de entorno).
- Una abstracción mínima para guardar respuestas raw en disco.
- Un pipeline de extracción histórica reproducible e idempotente, que
  descubre meetings y sesiones desde la API y registra cada ejecución en un
  manifest (ver [Pipeline de extracción histórica](#9-pipeline-de-extracción-histórica)).

No se implementan LSTM/GRU, métodos estadísticos avanzados, FastAPI,
Kubernetes, ni entrenamiento real sin autorización.

La arquitectura, configuración, comandos, validación sintética y pendientes
de M7.1 están documentados en [M07_TRAINING_MLFLOW.md](M07_TRAINING_MLFLOW.md).
La publicación inmutable de versiones registradas de M7 y su carga verificada
desde M8 se describe en [M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md).

> **Actualización (preparación cloud):** M1–M6 son *cloud-ready*: el
> almacenamiento está desacoplado (local o S3), cada stage se ejecuta como
> job batch independiente y existe una imagen Docker. El estado de diseño,
> código, validación local y despliegue se mantiene en
> [infra/aws_status.json](infra/aws_status.json); los estados de despliegue y
> pruebas reales son **NO VERIFICABLES** sin evidencia humana. Ver
> [Preparación para la nube](#14-preparación-para-la-nube-aws-ready).

## 2. Arquitectura

Flujo de dependencias, de arriba hacia abajo:

```
Configuration (config/settings.py)
        ↓
BaseAPIClient (ingestion/client.py)   -- rate limiting, retries, timeouts, errores, logging
        ↓
F1Client (ingestion/openf1.py)        -- métodos por endpoint de OpenF1
        ↓
RawDataStorage (storage/raw_storage.py)  -- persistencia de respuestas raw
```

Sobre esa base, el módulo de pipelines orquesta *qué* se descarga:

```
config.yaml (historical_extraction)
        ↓
HistoricalExtractionPipeline (pipelines/historical_extraction.py)
        ├── discover_meetings(year)     -- vía F1Client
        ├── discover_sessions(meeting)  -- vía F1Client
        ├── filter_sessions()           -- por tipo de sesión configurado
        ├── extract_endpoint()          -- vía F1Client
        ├── save / skip                 -- vía RawDataStorage
        └── ExecutionManifest           -- registro de la ejecución
```

La frontera es deliberada: `F1Client` responde *cómo* se hace una petición a
OpenF1; el pipeline responde *qué* datos hacen falta, para qué años y en qué
orden. El pipeline no reimplementa retries, rate limiting ni manejo de HTTP.

Y sobre los datos ya descargados, la validación juzga sin tocarlos:

```
data/raw + manifest
        ↓
ValidationRunner (validation/runner.py)
        ├── RawDataCatalog        -- descubrir y leer archivos raw
        ├── reglas por archivo    -- estructura, tipos, valores, nulos, duplicados, temporal
        ├── reglas de conjunto    -- identificadores, cobertura
        └── ValidationReport      -- PASS / WARNING / FAIL
        ↓
data/validation/validation_report_<timestamp>.json
```

`data/raw/` es inmutable: la validación nunca escribe en él.

Ningún otro módulo del proyecto debe hacer `requests.get(...)` /
`httpx.get(...)` directamente contra OpenF1: todo pasa por `F1Client`. Esto
mantiene la lógica de negocio (futuros notebooks, feature engineering, etc.)
completamente desacoplada del transporte HTTP.

**Separación de responsabilidades:**

| Componente | Responsabilidad | Por qué existe |
|---|---|---|
| `config/settings.py` | Cargar y validar configuración (YAML + env vars) | Nada debe leer `config.yaml` u `os.environ` directamente |
| `ingestion/rate_limiter.py` | Limitar frecuencia de llamadas (sliding window) | Respetar los límites de OpenF1 sin `time.sleep(1)` ingenuo |
| `ingestion/client.py` | HTTP genérico: rate limiting + retries + timeouts + errores + logging | Reutilizable para cualquier endpoint, sin duplicar lógica de transporte |
| `ingestion/exceptions.py` | Excepciones propias (`OpenF1*Error`) | El resto del código no debe depender de excepciones de `httpx` |
| `ingestion/models.py` | Helper `build_params` para filtros de query | Evita repetir "descartar `None`" en cada método `get_*` |
| `ingestion/openf1.py` | `F1Client`: un método por endpoint de OpenF1 | Único lugar que conoce las rutas/parámetros de OpenF1 |
| `storage/raw_storage.py` | `RawDataStorage.save(...)` a JSON particionado | Punto de extensión para el futuro pipeline de datos |
| `utils/logging.py` | Formatter JSON + `configure_logging()` | Logs estructurados y parseables sin dependencias externas |
| `pipelines/historical_extraction.py` | Recorrido año → meeting → session → endpoint | Decide *qué* descargar, sin conocer detalles de HTTP |
| `pipelines/manifest.py` | `ExecutionManifest`: registro de cada ejecución | Permite responder después "¿qué se descargó exactamente?" |
| `storage/raw_catalog.py` | Descubrir y leer los archivos de `data/raw` | El formato de rutas y del envelope se describe en un solo paquete |
| `validation/specs.py` | Qué se espera de cada endpoint, de forma declarativa | Ajustar una regla es editar datos, no lógica |
| `validation/validators.py` | Las comprobaciones, genéricas sobre las specs | Una sola pasada por archivo, sin importar cuántas reglas haya |
| `validation/runner.py` | Orquestar la validación y producir el reporte | Streaming: la memoria no depende del tamaño del dataset |
| `storage/backends/` | Escrituras que no pierden archivos y JSON idéntico byte a byte | Ni sobrescritura silenciosa, ni archivos a medio escribir, ni bytes que dependan del sistema operativo |

### Decisiones respecto a la estructura sugerida

Se mantuvo la estructura propuesta casi intacta, con dos adiciones menores:

- **`ingestion/rate_limiter.py`**: no estaba en el árbol original, pero el
  rate limiting es suficientemente complejo (dos ventanas simultáneas,
  thread-safety) como para merecer su propio módulo en vez de vivir inline
  dentro de `client.py`.
- **`scripts/check_connection.py`**: pequeño script ejecutable para la
  prueba de conexión end-to-end pedida en los entregables (no forma parte
  del paquete instalable).
- **`pipelines/`**: la orquestación vive en su propio paquete para que
  `F1Client` no acabe decidiendo qué años o sesiones descargar. Su
  `__init__.py` no re-exporta nada a propósito: un import ansioso haría que
  Python cargara dos veces el módulo al ejecutarlo con `python -m`.

## 3. Estructura del repositorio

```
f1-race-intelligence/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── Dockerfile                 # imagen batch M2–M6 (usuario sin privilegios)
├── .dockerignore
│
├── configs/
│   ├── config.yaml            # base (entorno local)
│   └── config.aws.yaml        # overlay para environment=aws (solo diferencias)
│
├── requirements/
│   ├── runtime-lock.txt       # runtime M1-M6 e imagen Docker
│   └── training-lock.txt      # cierre exacto para el extra M7
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── manifests/        # registro JSON de cada ejecución del pipeline
│   └── validation/       # reporte JSON de cada ejecución de validación
│
├── notebooks/
│
├── scripts/
│   └── check_connection.py
│
├── src/
│   └── f1_race_intelligence/
│       ├── __init__.py
│       ├── __main__.py            # python -m f1_race_intelligence (CLI batch)
│       ├── cli.py                 # --stage plan|m2|m3|m4|m5|m6|copy-raw
│       ├── scope.py               # SessionScope: unidad de trabajo (sesión)
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── client.py          # BaseAPIClient (HTTP genérico)
│       │   ├── exceptions.py
│       │   ├── models.py
│       │   ├── openf1.py          # F1Client
│       │   └── rate_limiter.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── historical_extraction.py   # HistoricalExtractionPipeline
│       │   └── manifest.py                # ExecutionManifest
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── backends/
│       │   │   ├── base.py                # StorageBackend (interfaz) + JSON/Parquet/joblib
│       │   │   ├── local.py               # LocalStorageBackend
│       │   │   └── s3.py                  # S3StorageBackend (boto3 opcional)
│       │   ├── layout.py                  # contrato de rutas: legacy | lake
│       │   ├── store.py                   # DataStore = backend + layout
│       │   ├── migration.py               # copia raw entre stores sin re-descargar
│       │   ├── raw_catalog.py             # lectura del árbol raw
│       │   └── raw_storage.py             # escritura del árbol raw
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── models.py                  # Severity, ValidationResult, ValidationReport
│       │   ├── specs.py                   # expectativas por endpoint (medidas, no supuestas)
│       │   ├── validators.py              # reglas
│       │   ├── manifest_index.py          # lectura del manifest de M2
│       │   ├── report.py                  # persistencia del reporte
│       │   └── runner.py                  # ValidationRunner
│       │
│       ├── consolidation/ features/ modeling/   # M4, M5, M6
│       │
│       ├── serving/                   # prototipo local de inferencia (M8, sección 19)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── files.py                   # Compatibilidad deprecada; usar storage/backends
│           └── logging.py                 # JSON a stdout + log_context(stage, year, session…)
│
└── tests/
    ├── unit/
    └── integration/
```

## 4. Instalación

Requiere Python 3.11+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

M7 es opcional: para entrenamiento local instala el lock específico y activa
el extra, sin añadir MLflow al runtime batch M1-M6:

```bash
python -m pip install -r requirements/training-lock.txt
python -m pip install -e ".[training]" --no-deps
```

Dependencias principales: `httpx` (HTTP), `tenacity` (retries con backoff),
`pydantic` (validación de configuración), `PyYAML`, `python-dotenv`.
Dependencias de desarrollo: `pytest`, `pytest-mock`, `respx` (mocking de
`httpx` para tests unitarios sin red).

## 5. Configuración

La configuración vive en [configs/config.yaml](configs/config.yaml):

```yaml
openf1:
  base_url: "https://api.openf1.org/v1"
  timeout:
    connect: 5
    read: 30
  rate_limit:
    requests_per_second: 3
    requests_per_minute: 30
  retry:
    max_attempts: 3
    backoff_factor: 1

logging:
  level: "INFO"
  json_format: true
```

Se carga con `f1_race_intelligence.config.settings.load_settings()`, que:

1. Lee `configs/config.yaml` (o la ruta indicada por `F1_CONFIG_PATH` o
   `--config`). **El archivo tiene que existir**: si falta, se detiene con
   `ConfigFileNotFoundError` (la CLI sale con código 2) indicando la ruta y
   cómo fijar `F1_CONFIG_PATH`, en vez de seguir con una configuración vacía.
   Un archivo vacío sí es válido y significa "todo por defecto". Además, `m2`
   y `plan` se niegan a arrancar (código 2) si no hay endpoints o temporada
   que procesar.
2. Si `environment` no es `local`, fusiona encima el overlay
   `configs/config.<environment>.yaml` (hoy: `config.aws.yaml`).
3. Carga un `.env` local si existe (ver [.env.example](.env.example)).
4. Aplica overrides puntuales por variable de entorno
   (`OPENF1_BASE_URL`, `OPENF1_RATE_LIMIT_RPS`, `OPENF1_RATE_LIMIT_RPM`,
   `OPENF1_RETRY_MAX_ATTEMPTS`, `LOG_LEVEL`, `F1_LOG_FILE`,
   `F1_ENVIRONMENT`, `F1_STORAGE_BACKEND`, `F1_STORAGE_LAYOUT`,
   `F1_STORAGE_LOCAL_ROOT`, `F1_STORAGE_BUCKET`, `F1_STORAGE_PREFIX`,
   `F1_STORAGE_REGION`, `F1_STORAGE_ENDPOINT_URL`).
5. Aplica los flags de la CLI (máxima prioridad).
6. Valida todo con `pydantic` y devuelve un `AppSettings` tipado.

Las secciones `environment` y `storage` se describen en
[Preparación para la nube](#14-preparación-para-la-nube-aws-ready). Sin
tocarlas, todo funciona exactamente como antes: archivos locales bajo `data/`.

```python
from f1_race_intelligence.config.settings import load_settings

settings = load_settings()
print(settings.openf1.base_url)
```

`max_attempts` es el número **total** de intentos (incluyendo el primero),
no el número de reintentos adicionales.

## 6. Prueba de conexión con OpenF1

```bash
python scripts/check_connection.py
```

Esto ejecuta el flujo completo:

```
OpenF1 API → F1Client → GET /sessions → respuesta procesada → log
```

Salida esperada (log JSON + resumen):

```
{"timestamp": "...", "level": "INFO", ..., "message": "openf1_request_success", "endpoint": "sessions", "status_code": 200, "duration_ms": 842.1}
OK: retrieved 23 race sessions from OpenF1 for 2023.
```

## 7. Uso de F1Client

```python
from f1_race_intelligence.ingestion.openf1 import F1Client

client = F1Client()

sessions = client.get_sessions(year=2025)
print(sessions)

laps = client.get_laps(session_key=9159, driver_number=1)
car_data = client.get_car_data(session_key=9159, driver_number=1)

client.close()
```

También puede usarse como context manager:

```python
with F1Client() as client:
    meetings = client.get_meetings(year=2024)
```

Métodos disponibles: `get_meetings`, `get_sessions`, `get_drivers`,
`get_laps`, `get_car_data`, `get_positions`, `get_intervals`, `get_stints`,
`get_pit`, `get_weather`, `get_race_control`. Todos aceptan filtros por
keyword y filtros adicionales vía `**kwargs` (se pasan como query params,
descartando los que sean `None`).

`get_car_data` es un caso especial: la telemetría se muestrea a ~3.7 Hz por
auto, así que una llamada sin filtrar sobre una sesión completa puede
devolver millones de filas en una sola respuesta. Por eso exige `driver_number`
y/o un filtro de rango de fechas (p. ej. `client.get_car_data(session_key=9159,
**{"date>": "2023-09-15T13:00:00", "date<": "2023-09-15T13:05:00"})`); si no
se da ninguno de los dos, lanza `ValueError` antes de hacer la petición.

### Guardar respuestas raw

```python
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.storage.raw_storage import RawDataStorage

client = F1Client()
storage = RawDataStorage()  # guarda bajo data/raw/ por defecto

params = {"year": 2025, "session_key": 9159}
data = client.get_laps(**params)
storage.save("laps", params, data)
# -> data/raw/laps/year=2025/session_key=9159/<timestamp>.json
```

### Manejo de errores

```python
from f1_race_intelligence.ingestion.exceptions import (
    OpenF1Error, OpenF1HTTPError, OpenF1RateLimitError,
    OpenF1TimeoutError, OpenF1ConnectionError,
)

try:
    client.get_sessions(year=2025)
except OpenF1RateLimitError:
    ...  # 429 tras agotar reintentos
except OpenF1TimeoutError:
    ...  # timeout de conexión o lectura
except OpenF1HTTPError as exc:
    ...  # otro error HTTP (usar exc.status_code)
except OpenF1Error:
    ...  # cualquier otro error del cliente
```

## 8. Rate limiting y retries

- **Rate limiting**: `SlidingWindowRateLimiter` mantiene, por ventana de
  tiempo (1s y 60s), las marcas de tiempo de llamadas recientes y bloquea
  lo estrictamente necesario para no exceder ningún límite configurado
  simultáneamente. Es thread-safe. No requiere librerías externas.
- **Retries**: implementados con `tenacity`, con backoff exponencial,
  número máximo de intentos configurable, y solo para errores transitorios
  (`429`, `500`, `502`, `503`, `504`, timeouts y errores de conexión). Un
  `404` o `400`, por ejemplo, falla inmediatamente sin reintentar. Si un
  `429` trae header `Retry-After`, ese valor prevalece sobre el backoff
  exponencial cuando es mayor (nunca esperamos menos de lo que pide el
  servidor). Cada reintento vuelve a pasar por el rate limiter — nunca se
  salta su `acquire()`.

## 9. Pipeline de extracción histórica

Descarga reproducible de datos históricos, orquestada desde configuración:

```bash
python -m f1_race_intelligence.pipelines.historical_extraction
```

El recorrido es `año → meetings → sessions → (filtro por tipo) → endpoints`,
y todo lo que descarga se define en `configs/config.yaml`:

```yaml
historical_extraction:
  years: [2023, 2024, 2025]
  session_types: ["Race"]        # también "Sprint", "Qualifying", "Practice"...
  endpoints: [drivers, laps, car_data, position, intervals, stints, pit, weather, race_control]
  output_path: "data/raw"
  manifest_path: "data/manifests"
  overwrite: false
```

Overrides por variable de entorno: `F1_EXTRACTION_YEARS` (lista separada por
comas), `F1_EXTRACTION_OVERWRITE`, `F1_EXTRACTION_OUTPUT_PATH`.

**Filtrado de sesiones.** `session_types` se compara sin distinguir
mayúsculas contra `session_name` *o* `session_type` de OpenF1, así que tanto
`"Race"` como `"Sprint Qualifying"` o un `"Practice"` más grueso (que agrupa
Practice 1/2/3) seleccionan lo esperado. Una lista vacía conserva todas las
sesiones. Nada en el código está atado a "Race".

**Idempotencia.** Cada archivo raw tiene una ruta determinística
(`<endpoint>/year=…/meeting_key=…/session_key=…/session_<key>.json`). Con
`overwrite: false` una segunda ejecución detecta lo ya descargado y lo omite
*sin* gastar cuota de rate limit; con `overwrite: true` vuelve a descargar
todo. Los logs registran cada decisión (`raw_file_downloaded`,
`raw_file_skipped`, `extraction_failed`).

Para que "el archivo existe" signifique de verdad "ese dato ya está
descargado", `RawDataStorage` escribe primero a un archivo temporal y luego
lo mueve a su nombre definitivo: una ejecución interrumpida nunca deja un
archivo truncado que la siguiente daría por bueno.

**Tolerancia a fallos.** Un error en un endpoint no detiene la sesión, uno
en una sesión no detiene el meeting, y uno en el descubrimiento de un año no
detiene los demás años. Todo queda registrado en el manifest.

**Manifest.** Cada ejecución escribe `data/manifests/manifest_<timestamp>.json`
con la configuración usada, meetings y sesiones procesados, una entrada por
archivo (estado, ruta, número de registros) y la lista de errores:

```json
{
  "pipeline_name": "historical_extraction",
  "pipeline_version": "1.0.0",
  "duration_seconds": 1.185,
  "config": { "years": [2023], "session_types": ["Race"], "overwrite": false },
  "summary": { "downloaded": 4, "skipped": 0, "failed": 0, "meetings": 1, "sessions": 1 },
  "meetings_processed": [1141],
  "sessions_processed": [7953],
  "entries": [
    { "endpoint": "stints", "status": "downloaded", "year": 2023,
      "meeting_key": 1141, "session_key": 7953, "record_count": 70,
      "file_path": "data/raw/stints/year=2023/meeting_key=1141/session_key=7953/session_7953.json" }
  ],
  "errors": []
}
```

**`car_data` es un caso especial.** `F1Client.get_car_data` rechaza consultas
sin `driver_number` (una sesión completa son millones de filas), así que el
pipeline descubre primero los pilotos de la sesión y guarda un archivo por
piloto (`driver_<n>.json`). Si el archivo raw de `drivers` ya existe, lo lee
de disco en vez de volver a consultar la API.

## 10. Validación de datos

Comprueba la calidad de lo que M2 descargó. **Nunca modifica `data/raw/`**:
lee, juzga y escribe un reporte. La limpieza pertenece a etapas posteriores.

```bash
python -m f1_race_intelligence.validation.runner
```

El recorrido es: descubrir archivos → analizar cada uno en **una sola pasada**
→ acumular agregados → reglas entre archivos → reporte en
`data/validation/validation_report_<timestamp>.json`.

### Reglas

| regla | qué comprueba |
|---|---|
| `structure` | JSON válido, payload en forma de lista, archivos vacíos |
| `schema` | campos obligatorios presentes; campos nuevos no descritos |
| `types` | tipos reales por campo (un `bool` no pasa como `int`) |
| `values` | rangos imposibles / implausibles / meramente inesperados |
| `missing_values` | conteo y proporción de nulos (no elimina nada) |
| `duplicates` | unicidad sobre la clave natural del endpoint |
| `temporal` | timestamps parseables y orden esperado |
| `identifiers` | referencias entre meetings, sessions, drivers y laps |
| `coverage` | qué falta y **por qué** falta |

### Severidades

Las tres se reparten según lo que los datos permiten afirmar:

- **ERROR** — imposible: velocidad negativa, duplicado sobre una clave
  natural verificada, JSON inválido. Hace que la ejecución sea `FAIL`.
- **WARNING** — implausible o incompleto de forma accionable.
- **INFO** — inesperado pero legítimo; se registra para que sea visible.

Los rangos no son inventados: salen de medir respuestas reales de OpenF1 y
están justificados junto a cada campo en
[validation/specs.py](src/f1_race_intelligence/validation/specs.py). Tres
ejemplos de por qué eso importa:

- `car_data.throttle` y `brake` alcanzan **104** aunque estén documentados
  como 0–100 → INFO, no error.
- `car_data.drs` es un **código de estado** (`{0,1,2,3,8,10,12,14}`), no un
  booleano.
- `intervals.gap_to_leader` es `float`, `None` **o** texto (`"+1 LAP"`) para
  los coches doblados: el 10% de los registros de una carrera. Una regla
  "debe ser numérico" marcaría miles de registros correctos.

### Por qué falta un dato

OpenF1 responde **HTTP 404 con `{"detail": "No results found."}`** cuando un
endpoint simplemente no tiene datos. M3 lee el manifest de M2 y usa el
`status_code` para distinguir:

| situación | severidad |
|---|---|
| 404 — la fuente no tiene ese dato (p. ej. `pit` en todo 2023) | INFO: `"pit data unavailable for 2023 in source"` |
| 5xx, 429 o timeout — la extracción falló | WARNING: reejecutar puede recuperarlo |
| nunca se intentó | WARNING |
| sesión del catálogo fuera de `session_types` | INFO: no se pidió |

### Muestreo

`max_records_per_file: null` (por defecto) valida **todos** los registros y es
el único modo cuyo resultado describe el dataset. Al fijar un valor N, la
ejecución pasa a ser una comprobación rápida determinista y el reporte lo
declara explícitamente: `sampling.exhaustive: false`, cuántos registros se
inspeccionaron y un aviso de que el resultado no es exhaustivo. Además se
emite un WARNING, de modo que **una ejecución muestreada nunca puede
aparecer como limpia** ni confundirse con la validación oficial.

## 11. Consolidación del dataset

Convierte los endpoints raw validados en un dataset analítico con una fila
por **piloto × carrera × vuelta**. No modifica `data/raw/`.

```bash
python -m f1_race_intelligence.consolidation.runner
```

Salida: `data/processed/lap_dataset/<año>/session_<key>.parquet`, más un
reporte de ejecución en `data/consolidation/`.

### El problema y cómo se resuelve

Los endpoints tienen granularidades muy distintas: `laps` da una fila por
vuelta, pero `car_data` da ~368 muestras por vuelta y `weather` una lectura
por minuto. Un join directo multiplicaría filas.

La solución se apoya en un hecho **medido**: para un piloto, las vueltas
teselan el reloj exactamente. Tomando la ventana de la vuelta t como
`[date_start(t), date_start(t+1))` y contrastándola contra `lap_duration` en
2.184 vueltas reales, salieron **0 huecos y 0 solapamientos**. Por eso cada
fuente de alta frecuencia se **reduce a grano de vuelta antes** de unirse, y
todos los joins son left joins sobre un grano que ya existe. La explosión de
cardinalidad es imposible por construcción, no por vigilancia.

| fuente | granularidad real | cómo se incorpora |
|---|---|---|
| `laps` | driver × lap | **base** |
| `car_data` | driver × ~3,7 Hz | agregados dentro de la ventana |
| `intervals` | driver × ~2 s | valor al final de la vuelta |
| `position` | driver × **cambio** | último valor conocido (solo el 24% de vueltas tiene registro) |
| `stints` | driver × stint | join por rango de vueltas |
| `pit` | driver × parada | `lap_number` es la vuelta de **entrada** |
| `weather` | session × 60 s | `merge_asof` **hacia atrás** |
| `race_control` | session × evento | conteos por vuelta |

### Regla anti-leakage

**Todo dato de la vuelta t viene de `[date_start(t), date_start(t+1))` o de
antes.** Nunca de t+1.

Por eso weather usa `merge_asof` hacia atrás y no *nearest*: la lectura más
cercana puede estar hasta 30 s en el futuro. La columna `weather_age_s` deja
la decisión auditable — en datos reales va de 0 a 60 s, nunca negativa.

M4 **no construye el target**. Deja `lap_duration` ordenado y `lap_number`
contiguo para que M5 haga el `shift(-1)` por piloto, y marca
`is_last_lap_for_driver` para que no genere un target inválido en abandonos.

### Muestras físicamente imposibles

La telemetría real trae valores imposibles (`n_gear` hasta 49, `throttle` y
`brake` a 104). Se **excluyen del agregado que contaminarían, campo por
campo**, nunca del archivo raw, y se cuentan en `car_data_invalid_samples` y
en el reporte. Un sensor de marcha roto no invalida la velocidad registrada
en la misma muestra.

### Trazabilidad y pérdida de registros

El reporte registra, por sesión y **por etapa**, cuántas filas entraron,
cuántas salieron y cuántas no encontraron correspondencia:

```
  stage              in    out  lost  matched  unmatched
  lap_windows      1058   1058     0     1055          3
  stints           1058   1058     0     1055          3
  intervals        1058   1058     0     1054          4
  car_data         1058   1058     0     1055          3
```

Las vueltas sin ventana cerrable (la última de quien abandona) **se
conservan** con los agregados nulos y marcadas: descartarlas sesgaría el
dataset hacia quienes terminaron.

### Escalabilidad

La telemetría de una sola carrera son ~180 MB y el histórico varios GB, así
que se procesa **un archivo de piloto a la vez** — la partición que ya
produjo M2. El pico de memoria medido es de **59 MB** consolidando dos
carreras completas (1,4 millones de muestras).

## 12. Features y target

Convierte el dataset consolidado en un dataset de modelado, con el target
`next_lap_time` y las variables históricas del piloto hasta la vuelta actual.

```bash
python -m f1_race_intelligence.features.runner
```

Salida: `data/features/lap_features/<año>/session_<key>.parquet` y un reporte
en `data/features/reports/`.

### Target

```
next_lap_time = lap_duration(t+1)
  groupby(session_key, driver_number) → sort by lap_number → shift(-1)
```

Solo empareja vueltas **realmente consecutivas**: un hueco en la numeración
no produce target, en vez de emparejar la vuelta 5 con la 7. Las filas sin
target válido —la última vuelta de cada piloto— **se conservan** con
`has_target = false`; nada se descarta en silencio.

### Regla de leakage

Ninguna feature puede usar información posterior a t. El único `shift(-1)`
del módulo construye el target.

**Caso encontrado en los datos reales:** `pit_duration` correlaciona **0,989**
con el exceso de tiempo de la vuelta t+1. Al revisarlo: de 43 paradas,
**ninguna** tiene su timestamp dentro de la vuelta t — la mediana está 23,1 s
*después* de que la vuelta terminara. La parada ocurre durante t+1, así que
esa columna es prácticamente una medición del target. Está excluida y
registrada en la lista de columnas prohibidas.

En cambio `pit_in_lap` **sí** es feature: la entrada al pit lane sucede antes
de cruzar la línea que cierra la vuelta t, por lo que es observable al
predecir. La diferencia está documentada en el catálogo.

### Catálogo de features

[features/selection.py](src/f1_race_intelligence/features/selection.py) es la
fuente de verdad: cada columna lleva un rol y la razón de su decisión.

| rol | significado |
|---|---|
| `feature` | segura: describe la vuelta t, completa al predecir |
| `review` | incluida, con reserva (nulos altos, redundancia, poca varianza) |
| `leaky` | excluida por contener información posterior a t |
| `excluded` | excluida por otras razones |
| `identifier` | trazabilidad, nunca feature |

Variables temporales, todas dentro de `(session_key, driver_number)` ordenado
por `lap_number` y con ventana **trailing que incluye t**: `lap_time_prev_1`,
`lap_time_delta_1`, `lap_time_roll_{mean,std,min}_{3,5}`,
`lap_time_vs_roll_mean_5` y `lap_time_expanding_mean`.

En la vuelta 1: `prev`, `delta` y las desviaciones quedan nulas; las medias y
mínimos valen la propia vuelta. Son nulos estructurales, no imputados.

### Validaciones anti-leakage

Siete guardas, seis bloqueantes. Las dos más importantes **recomputan** el
valor por un camino distinto al que lo produjo:

| guarda | qué comprueba |
|---|---|
| `grain_unique` | una fila por sesión, piloto y vuelta |
| `target_matches_independent_derivation` | el target reconstruido con un diccionario, no con el shift |
| `last_lap_has_no_target` | no se inventó target en la última vuelta |
| `target_does_not_cross_driver_or_session` | el target viene del mismo piloto y carrera |
| `no_forbidden_columns_among_features` | ninguna columna prohibida llegó al modelo |
| `rolling_features_use_only_the_past` | cada ventana recalculada solo con vueltas ≤ t |
| `no_feature_almost_equals_the_target` | **diagnóstico**, no bloquea: avisa si algo correlaciona > 0,99 |

Una sesión que falle una guarda bloqueante **no se escribe**: publicar un
dataset que se sabe contaminado es peor que no publicarlo.

### Lo que M5 no hace

No escala, no imputa y no codifica categóricas. Todo eso aprende parámetros
de los datos y debe ajustarse dentro del pipeline de entrenamiento, sobre el
split de train únicamente. `compound`, `team_name` y `driver_acronym` quedan
como `category` de pandas.

## 13. Dataset de modelado

Convierte el dataset de features en artefactos `train` / `validation` / `test`
listos para entrenar. **No entrena ningún modelo.**

```bash
python -m f1_race_intelligence.modeling.runner
```

Salida en `data/modeling/`, versionada bajo `runs/<run_id>/` dentro de cada
prefijo de split y de preprocessing. El reporte queda en
`data/modeling/reports/`.

### Orden de las operaciones

```
M5 → descartar filas sin target → split cronológico por carrera
   → ajustar preprocessing SOLO con train → transformar cada split → validar → escribir
```

Ese orden es el módulo entero. Ajustar el preprocessing **antes** del split,
sobre todos los datos, es la forma más común de que un modelo de tiempos de
vuelta parezca mejor de lo que es.

### Split temporal, por carrera

El split respeta el tiempo: entrenar con carreras antiguas y validar con
posteriores es lo único que dice algo sobre el futuro que se quiere predecir.

Y la unidad es la **carrera completa, nunca la vuelta**. Las vueltas de una
misma carrera comparten circuito, clima, safety cars y asignación de
neumáticos; repartirlas entre train y validation filtra la respuesta por las
circunstancias.

Dos estrategias, ambas configurables sin tocar código:

| estrategia | cómo asigna |
|---|---|
| `fraction` | ordena las carreras por fecha y corta por proporción |
| `years` | asigna temporadas completas a cada split |

### Preprocessing — ajustado solo con train

| paso | estrategia |
|---|---|
| imputación numérica | mediana **de train** (resiste los outliers de boxes y safety car) |
| indicadores de nulo | se conserva una columna que registra que el valor faltaba |
| escalado | estándar, con media y desviación **de train** (configurable) |
| categóricas | one-hot con `handle_unknown="ignore"` |

Los nulos aquí significan cosas distintas: un coche doblado no tiene gap
numérico, una primera vuelta no tiene vuelta anterior, y una trampa de
velocidad a veces no reporta. Ninguno es un cero, así que se imputan para que
un modelo pueda consumirlos **y** se conserva el indicador para no borrar el
hecho de la ausencia.

`handle_unknown="ignore"` importa: un piloto que debuta la temporada
siguiente aparece en validation sin haber estado en train, y debe codificarse
como ceros en vez de romper el pipeline.

### Guardas anti-leakage

Las comprobaciones de M6 incluyen la completitud de los tres splits y las
guardas anti-leakage existentes. Las tres centrales funcionan
**re-ajustando**: se entrena un transformador nuevo solo con train y otro con
train + validation, y el que está en uso debe coincidir con el primero y
diferir del segundo. Una guarda que solo inspeccionara nuestra intención
estaría de acuerdo con nosotros incluso estando equivocados.

Si la comparación no puede distinguir ambos ajustes, se reporta como
**inconcluyente**, no como aprobada.

Un split vacío, una guarda bloqueante omitida o un resultado inconcluyente
impiden el veredicto válido y la publicación. Con `fail_on_leakage=false`,
los fallos de guardas sí pueden dejar artefactos para diagnóstico, pero el
veredicto sigue siendo `INVALID`/`INCOMPLETE`/`INCONCLUSIVE` y M6 sale con
código distinto de cero.

### Salida

```
data/modeling/
├── train/runs/<run_id>/{dataset,features}.parquet
├── validation/runs/<run_id>/… ; test/runs/<run_id>/…
├── preprocessing/runs/<run_id>/{preprocessor.joblib,feature_schema.json}
└── reports/
```

Las rutas de los prefijos configurados y de cada split se conservan; cada
ejecución escribe en su propio directorio para no sobrescribir artefactos
anteriores. El reporte contiene el `run_id`, SHA-256 de lock y configuración,
commit de origen, estado de limpieza del checkout, fingerprint de entradas,
estados de guardas, veredicto global y hashes de artefactos releídos. Si el
commit o la limpieza del checkout no pueden determinarse, o falta el lock,
el reporte deja el dato explícitamente ausente y M6 no informa éxito.
`verify-m6` permite comprobar
un reporte y sus artefactos sin reescribirlos:
`python -m f1_race_intelligence --stage verify-m6 --report <ruta>`.

Cada split se escribe dos veces: sin transformar, para poder rastrear
cualquier fila hasta M5 y de ahí al raw, y codificado, que es lo que lee un
modelo. Las claves llevan prefijo `key_` porque una columna puede ser clave
**y** feature — `lap_number` lo es, ya que el coche se aligera al consumir
combustible — y sin el prefijo la clave sobrescribiría a la feature.

Todo en `data/modeling/` está fuera de Git y se reproduce volviendo a
ejecutar el módulo.

### Huellas de contenido en el reporte

Una ruta dice dónde estaba un dato; un hash dice qué era. El reporte de M6
registra el SHA-256 de los bytes reales, de modo que cualquiera puede
volver a calcularlo y comprobarlo:

| campo | contenido |
|---|---|
| `schema_version` / `verdict` | versión 2 y estado global: `VALID`, `VALID_WITH_WARNINGS`, `INCOMPLETE`, `INCONCLUSIVE`, `INVALID` o `ABORTED` |
| `run_id` | identificador único que también versiona las rutas de artefactos |
| `source_commit` / `source_dirty` | revisión real de Git y estado de limpieza, si están disponibles |
| `dependency_lock_sha256` | SHA-256 de `requirements/runtime-lock.txt` |
| `config_fingerprint` | SHA-256 de la configuración M6 sin ubicaciones de storage |
| `source.inputs` | cada archivo de M5 leído: `input` (`session_<s>.parquet`), `year`, `session_key`, `sha256`, `size_bytes` |
| `source.input_fingerprint` | huella agregada de las entradas, `sha256:<hex>` (también en `summary`) |
| `source.input_fingerprint_scheme` | versión del algoritmo: `m6-input-fingerprint/v1` |
| `artifact_files` | cada objeto escrito: `artifact`, `split`, `path`, `sha256`, `size_bytes` |

- Cada entrada se lee **una sola vez**; se hashea y se parsea desde esos
  mismos bytes, así que la huella describe exactamente lo que se modeló. Los
  archivos cuyas filas descartan después los filtros `years`/`session_keys`
  también se leyeron, y figuran.
- `artifact_files` cubre los seis Parquet (`dataset.parquet` y
  `features.parquet` por split), `preprocessor.joblib` y
  `feature_schema*.json`. Cada hash se calcula releyendo el objeto del
  almacenamiento después de escribirlo. Con las mismas entradas y la misma
  configuración, los ocho hashes son iguales en disco local y en S3
  simulado en las pruebas ejecutadas en Windows. La paridad en Linux queda
  pendiente de ejecutarse en un entorno Linux real.
- `input_fingerprint` es el SHA-256 de este documento JSON canónico (claves
  ordenadas, sin espacios, `ensure_ascii`):
  `{"inputs": [{"input", "year", "session_key", "sha256"}, …], "scheme": "m6-input-fingerprint/v1"}`,
  con las entradas ordenadas por `year` (`null` primero), `session_key`,
  `input` y `sha256`. No incluye URIs, rutas, bucket, backend, fechas ni
  números de ejecución: los mismos archivos dan la misma huella en disco
  local y en S3, en cualquier orden de listado.
- Un run abortado (guarda bloqueante, train vacío, sin filas con target)
  conserva `source.inputs` de lo que sí leyó, y deja `artifacts` y
  `artifact_files` vacíos, porque no escribió nada. Si falla una lectura o
  una escritura del almacenamiento, el error se propaga y no se escribe
  reporte, como hasta ahora.
- Los reportes nuevos usan `schema_version: 2` y estados `PASS`, `FAIL`,
  `INCONCLUSIVE` o `SKIPPED` (con motivo). La CLI oficial solo devuelve éxito
  con veredicto `VALID` o `VALID_WITH_WARNINGS`. Reportes v1 se pueden verificar
  sin reescritura.
- La estrategia de split actual se conserva. La separación futura por
  `meeting_key`, el corte UTC del 1 de julio de 2025 y el tratamiento de Sprint
  dependen de la aprobación académica pendiente.

## 14. Preparación para la nube (AWS-ready)

> El estado de AWS se registra únicamente en
> [infra/aws_status.json](infra/aws_status.json). El diseño, el código y la
> validación local se distinguen del despliegue y de las pruebas reales; estos
> últimos permanecen **NO VERIFICABLES** sin comprobación humana.

La lógica funcional, estadística y anti-leakage de M4/M5/M6 **no cambió**.
Se verificó ejecutando el pipeline nuevo sobre los datos locales y
comparando contra las salidas generadas por el código anterior: los Parquet
de M4, M5 y M6 son idénticos, y el reporte de M3 coincide hallazgo por
hallazgo.

### 14.1 Arquitectura local (sin cambios de comportamiento)

```
configs/config.yaml ──► load_settings() ──► AppSettings
                                               │
      M2 extracción ─► M3 validación ─► M4 consolidación ─► M5 features ─► M6 modelado
             │               │                 │                 │              │
             └───────────────┴──── DataStore (backend + layout) ─┴──────────────┘
                                               │
                              LocalStorageBackend + layout "legacy"
                                               │
                        data/raw, data/processed, data/features, data/modeling…
```

`python -m f1_race_intelligence.<stage>.runner` sigue funcionando igual, con
las mismas rutas bajo `data/`.

### 14.2 Abstracción de storage

Ningún stage conoce el filesystem ni boto3; dependen de una interfaz:

```
Pipeline (M2–M6)
      ↓
DataStore ── layout: qué clave recibe cada dato (legacy | lake)
      ↓
StorageBackend ── exists · read/write bytes · list · write_if_absent
   ├── LocalStorageBackend   (archivos; escritura atómica tmp + replace)
   └── S3StorageBackend      (objetos; PutObject atómico, If-None-Match)
```

Sobre esas primitivas la interfaz ofrece JSON, Parquet, joblib y nombres
únicos para manifests/reportes. Garantías idénticas en ambos backends:

- Un objeto que existe está completo (la idempotencia depende de eso).
- Un manifest o reporte nunca sobrescribe a otro, ni con dos escritores
  simultáneos (local: creación exclusiva; S3: escritura condicional).
- Clave inexistente → `FileNotFoundError`; otro fallo → `StorageError`.
  Ambos son `OSError`, así que los manejadores existentes siguen valiendo.
- Parquet se lee siempre desde un buffer: pyarrow no puede inventar columnas
  a partir de directorios `year=…`.
- Un documento JSON tiene una sola representación binaria: una función lo
  serializa (UTF-8 sin BOM, sangría de dos espacios, saltos de línea LF, sin
  salto final) y ambos backends guardan esos bytes tal cual, siempre en modo
  binario. Las pruebas ejecutadas en Windows verifican la igualdad entre
  disco local y S3 simulado; la ejecución equivalente en Linux sigue
  pendiente. Los archivos escritos en Windows por versiones anteriores llevan
  CRLF: siguen siendo legibles y no se reescriben, pero su hash es distinto
  del que tendría el mismo documento escrito hoy.
- El plan local que escribe `cli.py:233` sigue fuera de este contrato y de los
  artefactos fingerprintados de M6. Usa modo texto y se tratará como una tarea
  independiente si pasa a requerir reproducibilidad de bytes.

`boto3` es opcional (`pip install ".[aws]"`, ≥ 1.35.2 por las escrituras
condicionales) y solo se importa al crear un cliente S3.

### 14.3 Contrato de rutas del Data Lake (layout `lake`)

```
s3://<bucket>/<prefix>/
├── raw/year=<y>/endpoint=meetings/meetings.json
├── raw/year=<y>/meeting=<m>/endpoint=sessions/sessions.json
├── raw/year=<y>/meeting=<m>/session=<s>/endpoint=<endpoint>/session_<s>.json
│                                         endpoint=car_data/driver_<n>.json
├── validated/year=<y>/session=<s>/validation_report_<ts>.json   (M3 por sesión)
├── validated/validation_report_<ts>.json                        (M3 global)
├── consolidated/year=<y>/session=<s>/session_<s>.parquet        (M4)
├── features/year=<y>/session=<s>/session_<s>.parquet            (M5)
├── modeling/split=<train|validation|test>/runs/<run_id>/{dataset,features}.parquet (M6)
├── modeling/preprocessing/runs/<run_id>/{preprocessor.joblib,feature_schema.json}
├── manifests/extraction/[year=<y>/session=<s>/]manifest_<ts>.json
├── manifests/{consolidation,features,modeling}/<stage>_report_<ts>.json
├── models/   (reservado para M7)
└── logs/     (reservado; los logs van a stdout)
```

Todo lo de una sesión comparte un prefijo: es la unidad de trabajo, de
permisos y de reproceso. `bucket` y `prefix` son configurables; los nombres
de capa son el contrato (ver `storage/layout.py`).

El layout se elige con `storage.layout`: `auto` (por defecto) usa `legacy`
en local —las rutas por stage de `config.yaml`, como siempre— y `lake` en
S3. También puede usarse `lake` sobre disco local para ensayar el árbol de
S3 sin AWS (`F1_STORAGE_LAYOUT=lake`, por defecto bajo `data/lake/`).

### 14.4 Configuración

```yaml
environment: local        # local | aws  (F1_ENVIRONMENT)

storage:
  backend: local          # local | s3   (F1_STORAGE_BACKEND)
  layout: auto            # auto | legacy | lake
  local_root: null
  bucket: null            # F1_STORAGE_BUCKET
  prefix: ""              # F1_STORAGE_PREFIX
  region: null
  endpoint_url: null      # solo LocalStack/MinIO
```

`environment: aws` fusiona [configs/config.aws.yaml](configs/config.aws.yaml)
(solo lo que cambia: `backend: s3`, `layout: lake`) y exige S3. El bucket
no está en ningún YAML: lo inyecta el job como `F1_STORAGE_BUCKET`.
**Las credenciales nunca van en YAML ni en `.env`**: en AWS se usa el rol
IAM de la tarea; en local, la cadena estándar de credenciales de AWS.

### 14.5 Ejecución batch por stage y por sesión

Un único entrypoint ejecuta cualquier stage, sobre todo lo configurado o
sobre una sesión:

```bash
# Listar las unidades de trabajo (sesiones) de una temporada
python -m f1_race_intelligence --stage plan --year 2024 --plan-output plan.json

# Una sesión, de extremo a extremo (M6 es global: necesita toda la historia)
python -m f1_race_intelligence --stage m2 --year 2024 --meeting-key 1229 --session-key 9472
python -m f1_race_intelligence --stage m3,m4,m5 --year 2024 --meeting-key 1229 --session-key 9472
python -m f1_race_intelligence --stage m6

# Mismo job contra S3 (cuando exista el bucket)
python -m f1_race_intelligence --stage m4 --year 2024 --session-key 9472 \
    --storage-backend s3 --bucket <bucket> --prefix dev

# Leer de un sitio y escribir en otro
python -m f1_race_intelligence --stage m4 --session-key 9472 \
    --input-storage s3://<bucket>/dev --output-storage ./data/smoke --output-layout lake

# Subir lo ya extraído al layout lake SIN volver a descargar de OpenF1
python -m f1_race_intelligence --stage copy-raw --input-layout legacy \
    --output-storage s3://<bucket>/dev
```

| stage | por sesión | notas |
|---|---|---|
| `plan` | — | descubre meetings/sesiones (catálogos persistidos) y lista unidades |
| `m2` | sí | idempotente; el meeting sale del catálogo guardado, sin llamadas extra |
| `m3` | sí | valida la sesión + catálogos; guarda el alcance en `source.filter` |
| `m4` | sí | lista solo el prefijo de la sesión; lee telemetría piloto a piloto |
| `m5` | sí | |
| `m6` | no | batch global; split 2023+2024 / 1.ª mitad 2025 / 2.ª mitad 2025 intacto |
| `copy-raw` | opcional | copia byte a byte raw y manifests entre stores |

Códigos de salida: `0` éxito · `1` stage fallido (reintentable: todo es
idempotente) · `2` argumentos o configuración inválidos. Un `404` de OpenF1
(dato inexistente en la fuente) no hace fallar M2; un 5xx/timeout sí.
`--strict` hace que un M3 con estado `FAIL` devuelva `1`.

Con jobs por sesión, cada stage elige los registros correctos: M3 usa el
manifest más reciente que procesó esa sesión (o combina los manifests por
sesión si no hay una ejecución global), y M4 solo acepta reportes de
validación globales o de su misma sesión.

### 14.6 Logging

JSON por línea a **stdout** (lo que recoge CloudWatch). Cada línea dentro de
un stage lleva `stage`, `year`, `meeting_key`, `session_key` cuando aplican;
los eventos añaden `endpoint`, `record_count`, `elapsed_ms`/`elapsed_seconds`,
`status`, `error_type` y, en Linux, `peak_memory_mb`. El archivo local es
opcional (`logging.file_path` / `F1_LOG_FILE`) y nada depende de él.

### 14.7 Docker

```bash
# Imagen trazable: solo desde un árbol limpio, con el SHA completo del commit
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    echo "Hay cambios sin commit: no se construye la imagen." >&2
    false
else
    docker build --platform linux/amd64 \
        --build-arg F1_GIT_COMMIT="$(git rev-parse HEAD)" \
        --build-arg F1_GIT_DIRTY=false \
        -t f1-race-intelligence:dev .
fi

# Prueba local sobre los datos del repo, escribiendo en data/smoke (layout lake)
docker run --rm -v "$PWD/data:/app/data" f1-race-intelligence:dev \
    --stage copy-raw --input-layout legacy --output-storage /app/data/smoke --output-layout lake
docker run --rm -v "$PWD/data:/app/data" \
    -e F1_STORAGE_LAYOUT=lake -e F1_STORAGE_LOCAL_ROOT=/app/data/smoke \
    f1-race-intelligence:dev --stage m3,m4,m5 --year 2024 --meeting-key 1229 --session-key 9472
```

- `python:3.12-slim`, dependencias exactas desde `requirements/runtime-lock.txt`
  + `pip check`, proyecto instalado con `--no-deps`.
- Usuario sin privilegios (`uid 10001`); solo `/app/data` es escribible. Con
  un volumen montado en Linux, añadir `--user "$(id -u):$(id -g)"` si el
  directorio del host no es escribible para ese uid.
- La receta solo construye desde un árbol limpio
  (`git status --porcelain --untracked-files=all` vacío: incluye los archivos
  sin seguimiento aunque `status.showUntrackedFiles` los oculte) e inyecta
  `F1_GIT_COMMIT` con el SHA completo (`git rev-parse HEAD`) y
  `F1_GIT_DIRTY=false`. Una imagen construida sin esos build args no es
  trazable: la imagen no incluye `.git/`,
  así que el reporte deja la revisión explícitamente nula; el hash del lock se
  calcula desde el archivo incluido en la imagen.
- Sin datasets, `.env` ni credenciales (`.dockerignore`).
- Configuración por variables de entorno (`F1_ENVIRONMENT`, `F1_STORAGE_*`).
- `F1_CONFIG_PATH` es explícito: el paquete instalado no puede ubicar
  `configs/` por su propia ruta. Si se borra o apunta a un archivo que no
  existe, el contenedor sale con código 2 en lugar de "terminar bien" sin
  procesar nada.

> Imagen construida y probada en local con Docker Engine (WSL, `linux/amd64`):
> `--help`, `pip check`, uid 10001, rutas de configuración inexistentes
> (código 2), configuración AWS/S3 sin red, y `copy-raw` + M3→M5 de una sesión
> sin red con `--memory 512m --cpus 0.5`. La publicación en ECR es
> **NO VERIFICABLE** desde el repositorio (ver
> [infra/aws_status.json](infra/aws_status.json)).

### 14.8 Infraestructura como código

La infraestructura base (S3, Lambda de disponibilidad de OpenF1, EventBridge,
Step Functions, SNS, CloudWatch, ECR, IAM) está definida con AWS SAM en
[infra/](infra/README.md). `infra/aws_status.json` es la fuente de verdad
versionada: el estado de despliegue y de pruebas en AWS real permanece
**NO VERIFICABLE** sin comprobación humana. La Fase 1B define M2–M6 como
tareas Fargate endurecidas (raíz de solo lectura, sin capacidades Linux,
usuario 10001, imagen fijada por digest) orquestadas por Step Functions.

### 14.9 Arquitectura AWS propuesta

```
EventBridge Scheduler
        │
Step Functions (orquestación)
  ├─ plan (Fargate task)  ──► lista de sesiones
  ├─ Map sesiones  (concurrencia M2 = 1–2: límite de OpenF1)
  │     └─ M2 (Fargate) ─► M3 ─► M4 ─► M5     reintentos por sesión
  └─ M6 (Fargate / Batch, más memoria)
        │
S3 data lake (versionado, cifrado SSE, lifecycle)   ECR (imagen)
CloudWatch Logs + alarmas                            IAM role por tarea (mínimo privilegio)
Glue Data Catalog / Athena (opcional, partition projection)
```

## 15. Modelo de carrera (M7): pronóstico en carrera desde la vuelta *k*

Dado todo lo observado hasta el final de la vuelta *k*, el modelo entrega la
matriz N×N `P[i, j] = Pr(piloto i termina j-ésimo)` más `Pr(DNF)` por piloto. Los
coches retirados ocupan las últimas posiciones, así que la matriz es doblemente
estocástica por construcción. La matemática y cada decisión están en
`docs/f1-model-math.md`; el alcance, en `docs/f1-model-scope.md`; la
especificación implementada (formulación, supuestos, censura y DNF, priors,
alcance y protocolo anti-leakage), en `docs/m7-model-spec.md`.

| Módulo (`race_model/`) | Qué hace |
|---|---|
| `racedata` | Lee una carrera del raw (local o S3) o en vivo de OpenF1; `truncate_at_lap` es la guarda anti-leakage: corta todo después de que el líder completa la vuelta *k* y elimina el resultado oficial |
| `regime` | Vuelta a vuelta: verde / SC / VSC / bandera roja desde `race_control` |
| `pace` | Nivel 1: vueltas limpias, ritmo θ̂ por carrera (ridge, SE por covarianza, AR(1)), degradación por compuesto |
| `season` | Nivel 2: forma de coche y piloto entre carreras (Kalman, ρ ≡ 1) + clasificación como medición |
| `events` | Riesgos de SC/VSC/roja y de abandono por vuelta (pooling por circuito), pérdida en boxes, momento de parada |
| `forecast` | Posterior del ritmo con las vueltas 1..*k* y Monte Carlo vectorizado → matriz |
| `scoring`, `baselines` | RPS, log-loss por pares, calibración; baselines "orden persiste + parada" y matriz de transición |
| `artifact`, `runner` | Versiones inmutables en `models/m7/`, puntero `latest.json`, etapas `m7-*` |
| `lambda_handler` | Entrada de la Lambda de pronóstico (bajo demanda o en vivo) |

```bash
python -m f1_race_intelligence --stage m7-train,m7-evaluate --promote     # entrenar y promover si supera al baseline
python -m f1_race_intelligence --stage m7-forecast --year 2025 --session-key 9998 --lap 30
python -m f1_race_intelligence --stage m7-update                          # tras cada GP: estado de temporada al día
```

Split por fechas fijas: 2023–24 entrenamiento; 2025 hasta Spa validación; de
Budapest a Abu Dabi test. **El test está sellado** (`race_model.test_sealed`):
ninguna etapa lee sus carreras, salvo `m7-evaluate --split test --unseal-test`,
que queda registrado en `test_ledger.json` y se rechaza una segunda vez. Las
iteraciones del modelo se registran en `docs/m7-experiments.md`. Despliegue y mantenimiento en AWS: `infra/README.md`, sección
"Modelo de carrera (M7)".

## 16. Ejecutar tests

```bash
# Unit tests (sin red, con mocks) — se ejecutan por defecto
pytest

# Tests de integración (llaman a la API real de OpenF1), excluidos por defecto
pytest -m integration
```

Cobertura actual: construcción de URLs/params, rate limiting (ventanas
por segundo y por minuto), retries con backoff, timeouts, traducción de
errores HTTP a excepciones propias, carga/override de configuración,
almacenamiento raw, el pipeline de extracción histórica (descubrimiento,
filtrado, extracción, idempotencia, manifest y manejo de errores) y la
validación de datos (cada regla, severidades, estado global, reporte,
muestreo y tolerancia a archivos corruptos).

Preparación cloud: el contrato de `StorageBackend` se ejecuta idéntico sobre
disco y sobre un S3 simulado en memoria (sin boto3, red ni credenciales);
layouts `legacy`/`lake`; configuración local/AWS; M2–M6 sobre S3 comparados
contra su ejecución local; idempotencia; jobs por sesión; CLI y códigos de
salida; contexto de logging.

La cadena M1–M8 se puede recorrer de extremo a extremo, sin red y con datos
sintéticos, con la demostración descrita en [DEMO_LOCAL.md](DEMO_LOCAL.md)
(`python -m pytest scripts/demo_local_mlops.py -s`). Esa demostración no forma
parte de la suite por defecto.

## 17. Limitaciones actuales

- No hay modelos de datos tipados para las respuestas de OpenF1 (se
  devuelven como `list`/`dict` crudos).
- Raw sigue siendo un archivo JSON por respuesta (envelope); no hay todavía
  compresión ni formato columnar para la capa raw.
- La extracción respeta el límite de OpenF1 por cliente: varios jobs M2 en
  paralelo comparten ese límite, así que M2 no escala con más contenedores.
- `car_data` genera volúmenes grandes (~36k filas por piloto por carrera).
- **El despliegue AWS es NO VERIFICABLE** según
  [infra/aws_status.json](infra/aws_status.json); el backend S3 se ha probado
  contra un cliente simulado.
- CI con GitHub Actions (`.github/workflows/ci.yml`): en pull requests y
  pushes a `master`, revisa espacios en blanco y ejecuta
  `pytest -m "not integration"` (incluye `tests/infra`) en Ubuntu y Windows,
  y compara los hashes de M6 entre ambos. No ejecuta `cfn-lint` ni
  `sam validate`, no construye la imagen y no despliega (sin CD).

## 18. Licencia y uso de datos

Este proyecto consume la API pública [OpenF1](https://openf1.org/), cuyos
términos de uso la destinan a fines educativos, proyectos personales de
aprendizaje e investigación — encaja con el propósito de este trabajo de
Maestría en Ciencia de Datos y Analítica. No se distribuyen ni republican
los datos crudos obtenidos de OpenF1 fuera de este repositorio; los archivos
bajo `data/raw/` están excluidos de control de versiones (ver `.gitignore`).

## 19. Próximos pasos (fuera del alcance de esta etapa)

Ya implementado y demostrado **localmente** (con datos sintéticos): features
(M5), dataset de modelado (M6), entrenamiento con MLflow (M7), publicación del
modelo, serving FastAPI (M8), planificación incremental, backfill por
temporadas y protección de M6. Queda pendiente:

- Entrenamiento con datos reales de OpenF1 (requiere autorización) y
  evaluación académica de los resultados; modelos más complejos que la
  regresión lineal.
- Validación de la infraestructura AWS en una cuenta real (despliegue
  controlado, IAM y S3 reales, Step Functions, alarmas y SNS); hoy solo está
  validada estáticamente y por simulación.
- Construir y publicar la imagen Docker en ECR y registrar su digest.
- Una arquitectura AWS para M7 (MLflow remoto en lugar de SQLite local) y para
  servir M8 desplegado.
- Integración de R1–R3 en `master`, ya con auditoría independiente aprobada
  (ver la visión general); la validación en AWS real sigue pendiente.
- Medir la duración real de una temporada y del plan sobre el histórico
  completo, y definir una política de refresco de datos de OpenF1.
- CI/CD y despliegue automatizado.
- Dashboard de negocio y monitoreo de drift del modelo.

---

## 20. Prototipo de serving (M8)

> Prototipo **local**. No entrena modelos ni despliega en AWS/SageMaker. M7
> ya registra modelos localmente y la publicación los empaqueta (ver
> [M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md)); la cadena completa se
> demuestra con datos sintéticos en [DEMO_LOCAL.md](DEMO_LOCAL.md). Los valores
> de los ejemplos de esta sección son **ilustrativos**: no hay un modelo
> entrenado con datos reales.

### Arquitectura

```
POST /predict  {"records": [ {features de M5, sin transformar}, … ]}
  │
  ├─ serving/contract.py   valida contra feature_schema.json de M6:
  │                        nombres, tipos, ausentes, campos prohibidos
  ├─ M6: prepare_matrix + preprocessor.joblib (ajustado solo con train)
  │                        ← código y artefacto oficiales de M6, sin copias
  ├─ modelo cargado con mlflow.pyfunc (models:/… o runs:/…)  ← artefacto de M7
  └─ una predicción de next_lap_time por registro, en el orden recibido
```

| Módulo | Responsabilidad |
|---|---|
| `serving/contract.py` | Contrato de entrada, leído del `feature_schema.json` de M6 (ninguna lista de features escrita a mano) |
| `serving/model.py` | Interfaz `ModelLoader`; `MlflowPyfuncLoader` importa MLflow solo al cargar |
| `serving/service.py` | Recupera artefactos M7 por `runs:/` (o M6 local/`s3://`), comprueba identidad y hashes, y predice |
| `serving/app.py` | FastAPI: `GET /live` (liveness), `GET /health` (readiness), `GET /model-info`, `POST /predict` |

Al arrancar, el servicio puede recuperar `preprocessor.joblib`,
`feature_schema.json` y `data_provenance.json` del mismo run M7 que produjo el
modelo. Comprueba el `run_id` del modelo, los SHA-256 declarados por M7 y que
la firma enumere, en el mismo orden, las columnas codificadas de M6. La
referencia de preprocesamiento es `runs:/<M7_run_id>/m6`; la del modelo es la
versión exacta `models:/<nombre>/<versión>`. Si algo no concuerda, no acepta
peticiones. Un test exige que la matriz que recibe el modelo sea idéntica, en
train, validation y test, a la `features.parquet` que escribe el propio M6.

### Contrato de integración con M7

1. **Entrada:** un DataFrame con las columnas `encoded_feature_names` del
   `feature_schema.json` de la ejecución de M6 con la que se entrenó; es
   decir, `features.parquet` sin `next_lap_time` ni las columnas `key_*`.
2. **Salida:** un valor de `next_lap_time` por fila, en su escala original
   (la de `lap_duration`). Si M7 transforma el target, la inversa debe ir
   dentro del modelo registrado.
3. **Carga:** con `mlflow.pyfunc.load_model`, en cualquier flavor.
4. **Firma:** registrada con nombres de columna en el orden de
   `encoded_feature_names`. Si falta o no coincide, el servicio no está listo.
5. **Trazabilidad:** el modelo debe exponer su `run_id` M7. Los artefactos M6
   y `data_provenance.json` se recuperan desde ese mismo run; M7 enlaza el
   `m6_run_id`, `input_fingerprint` y hashes verificados.

Esta API no fija nombres de modelo registrado, alias ni experimentos: se
pasan en `F1_SERVING_MODEL_URI` cuando M7 los defina.

### Contrato de entrada

- **Features:** exactamente `input_features.numeric` + `input_features.categorical`
  del esquema de M6, sin transformar (con el catálogo actual de M5, hasta 51,
  de ellas 3 categóricas: `compound`, `driver_acronym`, `team_name`).
  `GET /model-info` devuelve la lista exacta.
- **Tipos:** numéricas, número finito, booleano o `null`; categóricas,
  texto o `null`.
- **Todas obligatorias como clave.** `null` significa desconocido: M6 lo
  imputa con la mediana de train y conserva el indicador de ausencia. Una
  categoría no vista en train se codifica como ceros.
- **Opcionales:** las columnas de trazabilidad (`year`, `meeting_key`,
  `session_key`, `driver_number`, `lap_number`, `session_date`). Se devuelven
  en `keys` y no llegan al modelo salvo que también sean features
  (`lap_number`).
- **Rechazados:** campos desconocidos y columnas prohibidas (el target y
  `FORBIDDEN_AS_FEATURES`, como `pit_duration`).
- **Lote:** de 1 a `F1_SERVING_MAX_RECORDS` registros (1000 por defecto).

### Ejecución local

```bash
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -e ".[training,serve]" --no-deps
export MLFLOW_TRACKING_URI=<tracking URI de M7>
export F1_SERVING_PREPROCESSING_URI=runs:/<run_id de M7>/m6
export F1_SERVING_MODEL_URI=models:/<nombre registrado por M7>/<versión exacta>
uvicorn f1_race_intelligence.serving.app:app --host 127.0.0.1 --port 8000
```

En PowerShell, las variables se definen con `$env:F1_SERVING_MODEL_URI = "..."`.
La documentación interactiva queda en `http://127.0.0.1:8000/docs`.

### Ejemplos (valores ilustrativos)

```http
GET /health
→ 200 {"status": "ok"}
→ 503 {"status": "unavailable", "message": "Model service is not ready.", "error_id": "<id>"}

GET /live
→ 200 {"status": "ok"}   # liveness is independent of model readiness
```

```http
POST /predict
{"records": [{"session_key": 9472, "driver_number": 1,
              "lap_number": 12, "lap_duration": 95.1, "tyre_age": 11,
              "compound": "MEDIUM", "gap_to_leader_s": null, …}]}

→ 200 {"target": "next_lap_time",
       "model": {"uri": "models:/<nombre>/<versión>", "run_id": "<run de M7>"},
       "predictions": [{"index": 0, "value": <predicción>,
                        "keys": {"session_key": 9472, "driver_number": 1, "lap_number": 12}}]}

→ 422 {"detail": [{"record": 0, "field": "next_lap_time",
                   "error": "this column must never reach the model (target or leakage)"},
                  {"record": 0, "field": "tyre_age",
                   "error": "missing feature: send null if the value is unknown"}]}
```

| Código | Cuándo |
|---|---|
| 200 | Predicción correcta |
| 422 | El cuerpo o algún registro incumple el contrato; se listan todos los problemas |
| 500 | El modelo devolvió algo que no es un valor finito por registro |
| 503 | El servicio no cargó (falta configuración, artefactos que no concuerdan, MLflow ausente…) |

### Tests

```bash
pytest tests/unit/test_serving.py        # contrato, artefactos, paridad con M6, loader MLflow
pytest tests/unit/test_serving_api.py    # capa HTTP; requiere el extra serve
```

Los tests generan artefactos auténticos de M6 ejecutando su runner sobre
carreras sintéticas. La integración M7–M8 también entrena con esos fixtures,
recupera el modelo por su versión exacta mediante pyfunc y verifica que los
artefactos y el modelo pertenecen al mismo run M7. No consulta OpenF1 ni usa
datos reales. CI instala los extras `training` y `serve` para ejecutar la
integración y la capa HTTP.

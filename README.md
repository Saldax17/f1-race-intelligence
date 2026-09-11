# F1 Race Intelligence

**F1 Race Intelligence** es un proyecto académico (Maestría en Ciencia de Datos y
Analítica) orientado a analizar y, en etapas futuras, predecir el desempeño de
pilotos de Fórmula 1 a partir de datos históricos de la API pública
[OpenF1](https://openf1.org/) (documentación: https://openf1.org/docs/).

La unidad de análisis objetivo es **Piloto × Carrera × Vuelta**, con la meta
futura de predecir `next_lap_time` (tiempo de la siguiente vuelta). Esa parte
predictiva **no** está implementada todavía — ver [Limitaciones actuales](#limitaciones-actuales).

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

No se implementa (todavía) Machine Learning, Deep Learning, feature
engineering, entrenamiento, MLflow, FastAPI, Docker, Kubernetes, AWS,
CI/CD ni dashboards.

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
| `utils/files.py` | Escrituras que no pierden archivos | Ni sobrescritura silenciosa ni archivos a medio escribir |

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
│
├── configs/
│   └── config.yaml
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
│       │   ├── raw_catalog.py             # lectura del árbol data/raw
│       │   └── raw_storage.py             # escritura del árbol data/raw
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
│       └── utils/
│           ├── __init__.py
│           └── logging.py
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

1. Lee `configs/config.yaml` (o la ruta indicada por `F1_CONFIG_PATH`).
2. Carga un `.env` local si existe (ver [.env.example](.env.example)).
3. Aplica overrides puntuales por variable de entorno
   (`OPENF1_BASE_URL`, `OPENF1_RATE_LIMIT_RPS`, `OPENF1_RATE_LIMIT_RPM`,
   `OPENF1_RETRY_MAX_ATTEMPTS`, `LOG_LEVEL`).
4. Valida todo con `pydantic` y devuelve un `AppSettings` tipado.

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

Salida en `data/modeling/`, con el reporte en `data/modeling/reports/`.

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

Doce comprobaciones, once bloqueantes. Las tres centrales funcionan
**re-ajustando**: se entrena un transformador nuevo solo con train y otro con
train + validation, y el que está en uso debe coincidir con el primero y
diferir del segundo. Una guarda que solo inspeccionara nuestra intención
estaría de acuerdo con nosotros incluso estando equivocados.

Si la comparación no puede distinguir ambos ajustes, se reporta como
**inconcluyente**, no como aprobada.

Un run que falle una guarda bloqueante **no escribe nada**.

### Salida

```
data/modeling/
├── train/dataset.parquet     # filas con trazabilidad, sin transformar
├── train/features.parquet    # matriz codificada + target + claves key_*
├── validation/ , test/       # idem
├── preprocessing/preprocessor.joblib   # el transformador ajustado con train
└── reports/
```

Cada split se escribe dos veces: sin transformar, para poder rastrear
cualquier fila hasta M5 y de ahí al raw, y codificado, que es lo que lee un
modelo. Las claves llevan prefijo `key_` porque una columna puede ser clave
**y** feature — `lap_number` lo es, ya que el coche se aligera al consumir
combustible — y sin el prefijo la clave sobrescribiría a la feature.

Todo en `data/modeling/` está fuera de Git y se reproduce volviendo a
ejecutar el módulo.

## 14. Ejecutar tests

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

## 15. Limitaciones actuales

- No hay modelos de datos tipados para las respuestas de OpenF1 (se
  devuelven como `list`/`dict` crudos).
- `RawDataStorage` es una abstracción mínima (un archivo JSON por
  respuesta); no es todavía la estrategia definitiva de almacenamiento.
- La extracción es secuencial (sin concurrencia): con el límite de 3 req/s
  de OpenF1, paralelizar aportaría poco y aumentaría el riesgo de `429`.
- `car_data` genera volúmenes grandes (~36k filas por piloto por carrera).
- La validación no consolida todavía los datos en un dataset: solo los juzga.
- No hay CI configurado.

## 16. Licencia y uso de datos

Este proyecto consume la API pública [OpenF1](https://openf1.org/), cuyos
términos de uso la destinan a fines educativos, proyectos personales de
aprendizaje e investigación — encaja con el propósito de este trabajo de
Maestría en Ciencia de Datos y Analítica. No se distribuyen ni republican
los datos crudos obtenidos de OpenF1 fuera de este repositorio; los archivos
bajo `data/raw/` están excluidos de control de versiones (ver `.gitignore`).

## 17. Próximos pasos (fuera del alcance de esta etapa)

- Feature engineering sobre la unidad Piloto × Carrera × Vuelta.
- Modelado y entrenamiento (baseline → modelos más complejos) para predecir
  `next_lap_time`.
- Tracking de experimentos con MLflow y model registry.
- Servir el modelo vía FastAPI.
- Containerización (Docker) y despliegue (Kubernetes / AWS).
- CI/CD.
- Dashboard y monitoreo de drift.

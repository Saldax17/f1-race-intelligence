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

### Decisiones respecto a la estructura sugerida

Se mantuvo la estructura propuesta casi intacta, con dos adiciones menores:

- **`ingestion/rate_limiter.py`**: no estaba en el árbol original, pero el
  rate limiting es suficientemente complejo (dos ventanas simultáneas,
  thread-safety) como para merecer su propio módulo en vez de vivir inline
  dentro de `client.py`.
- **`scripts/check_connection.py`**: pequeño script ejecutable para la
  prueba de conexión end-to-end pedida en los entregables (no forma parte
  del paquete instalable).

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
│   └── features/
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
│       ├── storage/
│       │   ├── __init__.py
│       │   └── raw_storage.py
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

## 9. Ejecutar tests

```bash
# Unit tests (sin red, con mocks vía respx) — se ejecutan por defecto
pytest

# Tests de integración (llaman a la API real de OpenF1), excluidos por defecto
pytest -m integration
```

Cobertura actual: construcción de URLs/params, rate limiting (ventanas
por segundo y por minuto), retries con backoff, timeouts, traducción de
errores HTTP a excepciones propias, y carga/override de configuración.

## 10. Limitaciones actuales

- No hay modelos de datos tipados para las respuestas de OpenF1 (se
  devuelven como `list`/`dict` crudos).
- `RawDataStorage` es una abstracción mínima (un archivo JSON por
  respuesta); no es todavía la estrategia definitiva de almacenamiento.
- No hay orquestación de pipeline (extracción histórica masiva por
  año/circuito), solo llamadas puntuales por endpoint.
- No hay CI configurado.

## 11. Licencia y uso de datos

Este proyecto consume la API pública [OpenF1](https://openf1.org/), cuyos
términos de uso la destinan a fines educativos, proyectos personales de
aprendizaje e investigación — encaja con el propósito de este trabajo de
Maestría en Ciencia de Datos y Analítica. No se distribuyen ni republican
los datos crudos obtenidos de OpenF1 fuera de este repositorio; los archivos
bajo `data/raw/` están excluidos de control de versiones (ver `.gitignore`).

## 12. Próximos pasos (fuera del alcance de esta etapa)

- Feature engineering sobre la unidad Piloto × Carrera × Vuelta.
- Modelado y entrenamiento (baseline → modelos más complejos) para predecir
  `next_lap_time`.
- Tracking de experimentos con MLflow y model registry.
- Servir el modelo vía FastAPI.
- Containerización (Docker) y despliegue (Kubernetes / AWS).
- CI/CD.
- Dashboard y monitoreo de drift.

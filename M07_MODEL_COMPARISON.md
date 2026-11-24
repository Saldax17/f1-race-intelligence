# M7 — Hoja de Comparación de Modelos: Modelo 1 (`race_model`) vs Ganador del Modelo 2 (`mlp_deeper_reg`)

## 1. Resumen ejecutivo

Esta hoja compara de manera directa y trazable los dos modelos desarrollados bajo la
especificación unificada de la etapa **M7** sobre datos reales de OpenF1 (temporadas
**2023–2025**):

- **Modelo 1 — Tradicional (`race_model`):** espacio de estados jerárquico
  (Nivel 1 Ridge + AR(1), Nivel 2 filtro de Kalman con $\rho \equiv 1$, riesgos parcial-pooled
  de SC/VSC/bandera roja/retiro) + simulación Monte Carlo vectorizada ($S = 4{,}000$).
  Especificación: [M07_RACE_MODEL.md](M07_RACE_MODEL.md).
  Artefacto verificado: versión `20261122T033003520681Z` (`verify-m7: OK`, promovido a
  `latest.json`, registrado en MLflow como `models:/f1-race-position-model/1`).
- **Ganador del Modelo 2 — Deep Learning (`mlp_deeper_reg`, rama `Sajimenezgo/DL`):**
  perceptrón multicapa tabular PyTorch CPU-only (`[128, 64, 32]`, 26 881 parámetros,
  dropout `0.20`, weight decay `5e-4`, pérdida Huber, `ReduceLROnPlateau` y *early stopping*),
  seleccionado estrictamente por `validation.mlp.mae` en `m7-compare` (`cmp-5089b42f`) frente
  a `mlp_default`, `mlp_small`, `mlp_mse`, `LinearRegression` (M7.1) y `DummyRegressor`.
  Registrado en MLflow como `models:/f1-next-lap-mlp/1` (junto al baseline lineal
  `models:/f1-next-lap-linear-regression/1`).

Ambos modelos cumplen hoy el **mismo contrato de ingeniería M7** (procedencia verificable,
`input_fingerprint`, empaquetado inmutable con `manifest.json` + marcador `PUBLISHED` SHA-256,
diagnóstico de generalización en 4 estados estrictamente sobre `train`/`validation`,
interpretabilidad estructurada, reporte académico en Markdown de 5 secciones, registro en el
mismo MLflow SQLite local y validación estricta de contratos de serving).

## 2. Comparación de especificación, contratos y arquitectura

| Dimensión | Modelo 1: `race_model` (ML Tradicional) | Ganador Modelo 2: `mlp_deeper_reg` (Deep Learning) |
|---|---|---|
| **Horizonte y objetivo** | Carrera completa desde vuelta *k*: matriz $N \times N$ doblemente estocástica $P_{ij} = \Pr(\text{piloto } i \text{ termina } j)$ + $\Pr(\text{retiro}_i)$ | Horizonte de 1 vuelta (`t → t+1`): tiempo escalar `next_lap_time` (segundos) por Piloto × Carrera × Vuelta |
| **Familia de modelo** | Espacio de estados jerárquico (Ridge + AR(1) + Kalman) + Monte Carlo vectorizado | MLP tabular PyTorch CPU-only (`Linear → ReLU → Dropout`, pérdida Huber) |
| **Datos de entrada** | 8 endpoints raw livianos (`drivers`, `laps`, `stints`, `pit`, `race_control`, `intervals`, `session_result`, `starting_grid`); **sin `car_data`** | Pipeline completo M4→M5→M6 (**56,1 M muestras de telemetría `car_data`** → 51 features → 128 columnas codificadas) |
| **Alcance de sesiones** | Solo Grandes Premios (`session_name == "Race"`), excluye Sprints y `2025 Miami Gardens` (vueltas 2–25 ausentes en origen) | 88 sesiones consolidadas de 2023–2025 (incluye Sprints); 84 140 filas brutas → 82 162 filas con target |
| **Splits y disciplina anti-snooping** | Fechas fijas: 46 train (2023–24), 12 val (2025 hasta Spa), **11 test (Budapest–Abu Dabi) selladas** (`test_sealed: true`, ledger de un solo uso) | Split cronológico M6: 54 407 train (2023–24), 13 449 val (2025 H1), 14 306 test (2025 H2, evaluado post-hoc; selección solo en val) |
| **Métrica principal** | RPS medio por piloto en pista (más log-loss por pares y Brier win/podium/points/DNF) | MAE y RMSE en segundos (más MAPE y $R^2$) |
| **Baselines** | `persist` (orden actual + parada pendiente) y `transition` (matriz empírica por fase) | `DummyRegressor(strategy="mean")` y `LinearRegression` (M7.1) |
| **Diagnóstico de generalización** | `balanced` (ratio RMSE prior val/train `0.99×`, calibración $z^2 = 0.98$, supera `persist`) | `balanced` (ratio MAE val/train `0.72×`, supera `DummyRegressor` por `−8.28 s`; *early stopped* en época 11/100, `best_epoch = 1`) |
| **Interpretabilidad** | **Estructural intrínseca** (`train`): $\phi$, $\sigma_\varepsilon$, pendientes de compuestos, descomposición de varianza Kalman, tasas de SC/retiro | **Post-hoc por permutación (PFI)** sobre `validation` (`n_rows=13 449`, 5 repeticiones) |
| **Integridad e inmutabilidad** | `version=<v>/{model.json, manifest.json, PUBLISHED}` + CLI `--stage verify-m7` | `published-models/.../{manifest.json, PUBLISHED}` + `--stage verify-m6` (8 artefactos SHA-256) |
| **Registro MLflow** | `models:/f1-race-position-model/1` (opcional/perezoso vía `--log-mlflow`, mismo SQLite) | `models:/f1-next-lap-mlp/1` (junto a `models:/f1-next-lap-linear-regression/1`) |
| **Tiempo local extremo a extremo** | **~103 s** directos desde raw (86 s train + 17 s evaluación) | **~20 min** M4–M6 (18,5 min consolidación `car_data`) + ~35 s por configuración MLP |
| **Dependencias y preparación AWS** | Solo `runtime-lock.txt` (`scipy`); **ECS Fargate + Step Functions + Lambda (`Dockerfile.forecast`) ya definidos en SAM** | Requiere extras `training` (MLflow) + `dl` (`torch==2.12.1+cpu`); prototipo local FastAPI M8 y MLflow SQLite |

## 3. Resultados medidos sobre datos reales (2023–2025)

### 3.1 Modelo 1 — `race_model` (validación: 12 carreras reales 2025 hasta Spa; `test` sellado)

- **Procedencia:** `version = 20261122T033003520681Z`,
  `input_fingerprint = sha256:f6230d8e0425c63d5bc25b092769fef506598f1dfbd3ae3cf7bd8a0490ecf4be`,
  `manifest_sha256 = f7f6a37e2a3944e57dcd3a63574578ddce6ee30d4ccf342d089b6755625fb92c`, `verify-m7: OK`.
- **Prior pre-carrera (`prior_check`):** `train` ($n = 754$) RMSE `0.4165`, $z^2 = 1.074$;
  `validation` ($n = 171$) RMSE `0.4131`, $z^2 = 0.981$.

| Fase (*k*) | `race_model` RPS | `persist` RPS | `transition` RPS | `race_model` Pairwise LL | `transition` Pairwise LL | Δ vs `persist` [IC 95%] | Δ vs `transition` [IC 95%] |
|---|---:|---:|---:|---:|---:|---|---|
| 10% | 0.1018 | 0.1597 | 0.0996 | 0.4466 | 0.4963 | −0.0579 [−0.0863, −0.0306] | +0.0022 [−0.0070, +0.0115] |
| 25% | 0.0980 | 0.1565 | 0.1055 | 0.4475 | 0.5258 | −0.0585 [−0.0877, −0.0306] | −0.0075 [−0.0267, +0.0100] |
| 50% | 0.0859 | 0.1365 | 0.0965 | 0.4080 | 0.4804 | −0.0506 [−0.0739, −0.0254] | −0.0106 [−0.0292, +0.0084] |
| 75% | 0.0662 | 0.0812 | 0.0649 | 0.3227 | 0.3686 | −0.0150 [−0.0318, +0.0002] | +0.0012 [−0.0086, +0.0100] |
| 90% | 0.0461 | 0.0462 | 0.0432 | 0.2088 | 0.2830 | −0.0001 [−0.0084, +0.0074] | +0.0029 [−0.0012, +0.0073] |
| **Global (`all`)** | **0.0796** | **0.1160** | **0.0819** | **0.3667** | **0.4308** | **−0.0364 [−0.0515, −0.0216]** | **−0.0024 [−0.0095, +0.0049]** |

- **Calibración de eventos (Brier):** `win = 0.0272`, `podium = 0.0464`, `points = 0.1131`,
  `dnf = 0.0390` (probabilidad media pronosticada `0.045` vs observada `0.041`).
- **Interpretabilidad estructural (Nivel 1 y Nivel 2 en `train`):**
  - Nivel 1: $\phi = 0.4145$, $\sigma_\varepsilon = 0.4134\%$ (multiplicador AR(1) de largo plazo `1.554×`); degradación por compuesto: `MEDIUM` `0.0662%/vuelta`, `SOFT` `0.0603%/vuelta` (offset `−0.1419%`), `HARD` `0.0573%/vuelta`.
  - Descomposición de varianza Nivel 2: ajuste equipo × circuito (`s_c`) **50.3%**, ruido transitorio del día de carrera (`s_w`) **44.5%**, evolución del equipo en temporada (`q_gamma`) **5.0%**, evolución de piloto (`q_delta`) **0.2%**. Peso de clasificación $\lambda = 0.9727$ ($\sigma_q = 0.8775\%$).

### 3.2 Ganador del Modelo 2 — `mlp_deeper_reg` vs candidatos y baselines M7 (`cmp-5089b42f`)

- **Procedencia M6:** `run_id = 327d9e73aa5042ac861733d89249253a`,
  `input_fingerprint = sha256:4c0ab2515e5d6a58488884305d5c8ee5754c8ef675636d5c8cc3f23f239ddcf0`,
  `verify-m6: OK` (8/8 artefactos SHA-256).

| Modelo / Configuración | Train MAE (s) | Train RMSE (s) | Val MAE (s) | Val RMSE (s) | Val MAPE (%) | Val $R^2$ | Test MAE (s) | Test RMSE (s) | Test MAPE (%) | Test $R^2$ | Diagnóstico |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline: `DummyRegressor(mean)` | 15.65 | 78.71 | 12.74 | 17.61 | 14.11 | −0.0378 | 13.81 | 49.73 | 13.53 | −0.0015 | baseline |
| Baseline: `LinearRegression` (M7.1) | 16.12 | 64.99 | 18.01 | 36.47 | 18.22 | −3.4513 | 17.10 | 47.75 | 16.72 | 0.0767 | baseline |
| **MLP: `mlp_deeper_reg` (ganador)** | **6.17** | **67.98** | **4.46** | **8.87** | **4.42** | **0.7365** | **4.81** | **45.46** | **3.46** | **0.1630** | **`balanced`** |
| MLP: `mlp_default` (`[64, 32]`, Huber) | 6.05 | 66.98 | 4.55 | 9.27 | — | 0.7126 | 4.90 | 45.39 | — | 0.1654 | `balanced` |
| MLP: `mlp_small` (`[32]`, Huber) | 6.44 | 67.88 | 5.03 | 11.62 | — | 0.5480 | 6.19 | 45.32 | — | 0.1683 | `balanced` |
| MLP: `mlp_mse` (`[64, 32]`, MSE) | 8.00 | 51.05 | 11.19 | 43.23 | — | −5.2569 | 9.05 | 63.49 | — | −0.6327 | `overfitting_risk` |

- **Interpretabilidad (Top 6 PFI en `validation` para `mlp_deeper_reg`, incremento medio de MAE ± std):**
  1. `duration_sector_3`: `+2.743 ± 0.014 s`
  2. `duration_sector_2`: `+2.039 ± 0.028 s`
  3. `lap_time_roll_min_3`: `+1.781 ± 0.023 s`
  4. `lap_time_roll_min_5`: `+1.587 ± 0.012 s`
  5. `duration_sector_1`: `+1.546 ± 0.021 s`
  6. `pit_in_lap`: `+0.511 ± 0.007 s`

## 4. Análisis crítico: fortalezas, brechas y complementariedad

1. **Dos horizontes complementarios (no intercambiables):**
   - `mlp_deeper_reg` sobresale en predicción táctica **vuelta a vuelta** (`t → t+1`): la pérdida Huber fue decisiva frente a MSE (`4.46 s` vs `11.19 s` en validación), reduciendo el MAE un **65%** respecto a `DummyRegressor` y un **75%** respecto a `LinearRegression`.
   - Sin embargo, `mlp_deeper_reg` **no puede simularse hasta la bandera a cuadros** (problema de simulabilidad: depende de sectores y telemetría de la vuelta *t* que no existen para vueltas futuras).
   - `race_model` resuelve la decisión estratégica de **carrera completa** produciendo distribuciones calibradas de posición final y eventos (`win`, `podium`, `points`, `DNF`), con manejo explícito de paradas en boxes, SC/VSC, banderas rojas y retiros.
2. **Sensibilidad a eventos extremos y telemetría:**
   - En `mlp_deeper_reg`, el RMSE salta de `8.87 s` en validación a `45.46 s` en test ($R^2$ cae de `0.7365` a `0.1630`) debido a picos de vueltas bajo bandera roja/SC en la segunda mitad de 2025, aunque Huber mantiene el MAE estable (`4.81 s`). Además, las 4 configuraciones MLP alcanzaron su mejor pérdida en `best_epoch = 1` (se sugiere evaluar menor tasa de aprendizaje inicial y filtrar Sprints fuera de M4–M6 en futuras iteraciones).
   - `race_model` no depende de `car_data` (~3.7 Hz, ausente en vivo en OpenF1 sin autenticación y responsable de ~18.5 min de consolidación local), entrena y evalúa en ~103 s, excluye Sprints y carreras incompletas (`2025 Miami Gardens`), y mantiene el conjunto `test` estrictamente sellado.
3. **Brechas abiertas en `race_model`:**
   - Supera de forma estadísticamente sólida a `persist` (`ΔRPS = −0.0364`, IC 95% `[−0.0515, −0.0216]`) y mejora a `transition` en RPS global (`0.0796` vs `0.0819`) y log-loss por pares (`0.3667` vs `0.4308`), pero con 12 carreras de validación el IC 95% frente a `transition` (`[−0.0095, +0.0049]`) cruza 0, y los buckets tardíos (`75%`, `90%`) ceden ligeramente frente a `transition` por el supuesto de adelantamiento sin fricción.

## 5. Procedencia de datos y destino de carga en AWS S3

- **Ejecución local actual:** los resultados de la Sección 3 se obtuvieron localmente leyendo el histórico OpenF1 2023–2025 en disco local sin modificarlo.
- **Bucket S3 destino para ejecución en la nube:** los datos del data lake se cargarán desde:
  - **`s3://f1-race-intelligence-dev-793264560810-us-east-1`** (`F1_ENVIRONMENT=aws`, `F1_STORAGE_BACKEND=s3`, `F1_STORAGE_LAYOUT=lake`).
- **Estado actual del bucket:** **los datos todavía no están cargados en ese bucket.** Siguiendo el contrato de [configs/config.aws.yaml](configs/config.aws.yaml) y `tests/unit/test_settings_environments.py`, el nombre del bucket **no se fija en YAML ni se incluyen credenciales en el repositorio**; se inyecta únicamente por variable de entorno (`F1_STORAGE_BUCKET`) o flag CLI (`--bucket` / `--input-storage`).

### Comandos preparados cuando el bucket esté poblado

```bash
# 1. Subir el histórico raw local al bucket en layout lake (sin re-descargar de OpenF1)
python -m f1_race_intelligence --stage copy-raw \
  --input-storage "<ruta-local-f1-data>" --input-layout legacy \
  --output-storage s3://f1-race-intelligence-dev-793264560810-us-east-1 --output-layout lake

# 2. Entrenar, evaluar, promover y verificar race_model directamente contra S3
export F1_ENVIRONMENT=aws
export F1_STORAGE_BUCKET=f1-race-intelligence-dev-793264560810-us-east-1
python -m f1_race_intelligence --stage m7-train,m7-evaluate,verify-m7 --promote

# 3. Opcional: leer raw desde S3 y registrar artefactos/reportes/MLflow en local
python -m f1_race_intelligence --stage m7-train,m7-evaluate,verify-m7 --promote --log-mlflow \
  --input-storage s3://f1-race-intelligence-dev-793264560810-us-east-1 --input-layout lake \
  --output-storage . --output-layout legacy
```

## 6. Honestidad académica y estado de cierre

- **Test sellado en Modelo 1 (`race_model`):** las 11 carreras de test (Budapest–Abu Dabi 2025) permanecen **sin abrir** (`test_sealed: true`, `test_ledger.json` vacío). Solo se evaluarán una vez para el cierre final mediante `m7-evaluate --split test --unseal-test`.
- **Anti-snooping en Modelo 2 (`mlp_deeper_reg`):** el ganador se eligió estrictamente por `validation.mlp.mae`; las métricas de `test` de la tabla 3.2 corresponden a evaluación post-hoc del barrido controlado `cmp-5089b42f`.
- **Estado AWS:** la carga de datos reales en `f1-race-intelligence-dev-793264560810-us-east-1` y el despliegue productivo de M7 en AWS continúan pendientes (ver [infra/aws_status.json](infra/aws_status.json)).

# M7 — Modelo de carrera tradicional (Espacio de Estados Jerárquico + Monte Carlo)

## Propósito y alcance académico

Este documento especifica el **modelo estadístico / de aprendizaje automático tradicional**
de la etapa M7 (`src/f1_race_intelligence/race_model/`), diseñado bajo los mismos contratos
de procedencia, anti-leakage, integridad inmutable, diagnóstico de generalización,
interpretabilidad, reporte académico, MLflow opcional y contrato de inferencia que los
modelos tabulares de M7 ([M07_TRAINING_MLFLOW.md](M07_TRAINING_MLFLOW.md) y el modelo Deep
Learning `TabularMLP`).

Mientras los modelos tabulares de M7 (`LinearRegression` y `TabularMLP`) predicen el tiempo
de la siguiente vuelta (`next_lap_time`) fila a fila sobre los artefactos de M6, el **modelo
de carrera (`race_model`)** responde a la pregunta probabilística a nivel de Gran Premio:
dado todo lo observado hasta el cierre de la vuelta *k*, entregar la matriz doblemente
estocástica $N \times N$ de posiciones finales:

$$P_{ij} = \Pr(\text{piloto } i \text{ termina en posición } j \mid \mathcal{F}_k), \qquad i, j = 1 \ldots N_r,$$

junto con la probabilidad marginal de retiro $\Pr(\text{retiro}_i \mid \mathcal{F}_k)$ por
piloto. Las derivaciones matemáticas completas viven en
[docs/f1-model-math.md](docs/f1-model-math.md), el historial de alcance en
[docs/f1-model-scope.md](docs/f1-model-scope.md), la especificación técnica concisa en
[docs/m7-model-spec.md](docs/m7-model-spec.md) y el registro de experimentos en
[docs/m7-experiments.md](docs/m7-experiments.md).

### Dos modelos bajo una misma especificación M7

| Dimensión | Modelo M7 Tradicional (`race_model`) | Modelo M7 Tabular / DL (`LinearRegression`, `TabularMLP`) |
|---|---|---|
| **Objetivo** | Matriz $N \times N$ de posición final $P_{ij}$ + $\Pr(\text{retiro}_i)$ desde vuelta *k* | Tiempo escalar `next_lap_time` (s) por Piloto × Carrera × Vuelta |
| **Familia** | Espacio de estados jerárquico (Ridge + AR(1) + Filtro de Kalman) + Monte Carlo | Regresión lineal / Perceptrón multicapa PyTorch CPU-only |
| **Entrada** | Raw validado de 8 endpoints livianos (`drivers`, `laps`, `stints`, `pit`, `race_control`, `intervals`, `session_result`, `starting_grid`; sin `car_data`) | Artefactos M6 verificados (`features.parquet`, `preprocessor.joblib`, `feature_schema.json`) |
| **Splits temporales** | Fechas fijas: 2023–24 train (46 carreras), 2025 hasta Spa val (12), Budapest–Abu Dabi test sellado (11) | Cronológico por carrera (M6): 2023–24 train, 1.ª mitad 2025 val, 2.ª mitad 2025 test |
| **Métrica principal** | RPS medio por piloto en pista (y log-loss por pares + Brier win/podium/points/DNF) | MAE y RMSE en segundos (más MAPE y $R^2$) |
| **Baselines** | `persist` (orden actual + parada pendiente) y `transition` (matriz empírica por fase) | `DummyRegressor(strategy="mean")` y `LinearRegression` |
| **Integridad inmutable** | `version=<v>/{model.json, manifest.json, PUBLISHED}` + CLI `--stage verify-m7` | `published-models/<nombre>/versions/<v>/{manifest.json, PUBLISHED}` + `--stage verify-m6` |
| **MLflow** | Opcional y perezoso (`--log-mlflow`, `models:/f1-race-position-model/<v>`) en el mismo SQLite | Tracking/Registry local (`f1-next-lap-linear-regression`, `f1-next-lap-mlp`) |
| **Serving** | CLI `m7-forecast` + AWS Lambda (`Dockerfile.forecast`), validado con contrato entrada/salida | FastAPI M8 (`/live`, `/health`, `/model-info`, `/predict`) con `serving/contract.py` |

### Por qué un modelo generativo jerárquico y no clasificación celda a celda

1. **Doble estocasticidad exacta por construcción:** estimar una matriz $N \times N$
   ($20 \times 20$ en 2023–25; $22 \times 22$ desde 2026) celda por celda rompería las
   restricciones de suma de columnas (solo un coche termina en cada puesto). Promediar
   matrices de permutación muestreadas por Monte Carlo garantiza filas y columnas que suman
   1 sin reparaciones post-hoc.
2. **Separación entre ritmo continuo y eventos discretos:** el ritmo limpio sigue una
   estructura lineal con errores AR(1) y forma latente de coche/piloto entre carreras,
   mientras que las paradas en boxes ($\approx 20\text{–}25\text{ s}$), safety cars, VSC,
   banderas rojas y abandonos son saltos discretos que rompen los modelos puramente
   regresivos sobre tiempos de vuelta.
3. **Interpretabilidad estructural intrínseca:** cada parámetro tiene significado físico y
   deportivo directo (pendiente de degradación por compuesto, persistencia AR(1), varianza
   de evolución del coche vs piloto vs circuito vs clasificación y riesgo de neutralización
   por circuito).

## Arquitectura del modelo e hiperparámetros

La implementación vive en `src/f1_race_intelligence/race_model/`, con configuración tipada
en `RaceModelSettings` ([src/f1_race_intelligence/config/settings.py](src/f1_race_intelligence/config/settings.py))
y valores declarativos bajo `race_model` en [configs/config.yaml](configs/config.yaml).

### Flujo de modelado e inferencia

```text
Datos raw hasta vuelta k (racedata.truncate_at_lap: corta > k y elimina resultado oficial)
  ├─ Nivel 1 (pace.py): filtro de vueltas limpias → Ridge (% vuelta mediana) + AR(1) (φ, σ_ε) + degradación por compuesto
  ├─ Nivel 2 (season.py): filtro de Kalman entre carreras (γ_coche + δ_piloto + c_circuito + w_carrera, ρ ≡ 1) + clasificación
  ├─ Eventos (events.py): riesgos parcial-pooled por circuito (SC, VSC, bandera roja, retiro) + pérdida/momento de parada
  └─ Pronóstico (forecast.py): posterior bayesiano del ritmo con vueltas 1..k → Monte Carlo vectorizado (S = 4 000)
        └─ Matriz N×N doblemente estocástica P[i, j] + Pr(retiro_i)  →  validada por contrato de salida
```

### Hiperparámetros y valores por defecto

| Parámetro (`race_model.*`) | Defecto | Restricción / Rol | Propósito |
|---|---|---|---|
| `years` / `train_years` | `[2023, 2024, 2025]` / `[2023, 2024]` | Temporadas enteras | Histórico total y temporadas de ajuste |
| `validation_until` / `test_until` | `"2025-08-01"` / `"2026-01-01"` | Fechas ISO fijas | Corte fijo: 2025 hasta Spa = val; Budapest–Abu Dabi = test |
| `test_sealed` | `true` | Booleano | Bloquea toda lectura del split `test` salvo `--unseal-test` |
| `exclude_races` | `["2025 Miami Gardens"]` | Lista de etiquetas | Excluye carreras con datos corruptos en origen (vueltas 2–25 faltantes) |
| `burn_in_races` | `6` | Entero $\ge 1$ | Carreras iniciales del filtro sin aporte a la verosimilitud |
| `free_air_seconds` | `1.5` | Flotante $> 0$ | Brecha mínima (s) al coche delantero para considerar vuelta limpia |
| `slow_lap_factor` | `1.04` | Flotante $> 1$ | Cota sobre la mediana de la carrera para filtrar vueltas lentas |
| `min_clean_laps` | `5` | Entero $\ge 1$ | Vueltas limpias mínimas por piloto para observación de ritmo |
| `ridge_sigma` | `0.4` | Flotante $> 0$ | Desviación de regularización Ridge en Nivel 1 (%) |
| `compound_offset_prior_sd` / `compound_slope_prior_sd` | `1.0` / `0.1` | Flotantes $> 0$ | Priors gaussianos débiles sobre offset (%) y degradación (%/vuelta) |
| `fit_rho` | `false` | Booleano | `false` mantiene caminata aleatoria ($\rho \equiv 1$, §3.3 xi) |
| `initial_team_variance` / `initial_driver_variance` | `0.25` / `0.05` | Flotantes $> 0$ | Priors de varianza inicial para equipo nuevo ($\gamma$) y debutante ($\delta$) |
| `quali_max_abs_pct` | `4.0` | Flotante $> 0$ | Recorte de tiempos de clasificación atípicos (%) |
| `k_fractions` | `[0.1, 0.25, 0.5, 0.75, 0.9]` | Fracciones en $(0, 1)$ | Momentos de carrera evaluados en validación/test |
| `scenarios` | `4000` | Entero $\ge 1$ | Escenarios Monte Carlo vectorizados (error estándar por celda $\le 0.5/\sqrt{S}$) |
| `restart_gap_seconds` | `0.8` | Flotante $> 0$ | Compresión de brecha por posición en relanzamiento tras SC/roja |
| `pit_under_sc_probability` / `pit_under_vsc_probability` | `0.7` / `0.4` | Probabilidades $[0, 1]$ | Probabilidad de tomar parada pendiente bajo neutralización |
| `noise_scales` / `noise_scale_per_phase` | `[1.0, 1.5, 2.0, 3.0]` / `true` | Grilla / booleano | Multiplicador de ruido de vuelta elegido en train (uno por fase *k*) |
| `pass_costs` | `[0.0, 1.0, 1.5, 2.0, 2.5, 3.0]` | Grilla (s/puesto) | Fricción de adelantamiento ($\text{s/puesto} \times \text{dificultad de circuito}$) elegida en `train` (sin telemetría) |
| `transition_shrinkage` | `true` | Booleano | Contracción convexa por fase $w_k \in [0, 1]$ hacia `TransitionBaseline.predict_running` con conteos LOO en `train` |
| `calibration_races` | `16` | Entero $\ge 1$ | Últimas carreras secas de `train` usadas para elegir multiplicador de ruido y fricción |
| `random_seed` | `42` | Entero | Semilla determinista compartida con el resto del proyecto |
| `promote_requires_beating` | `"persist"` | `"persist"` \| `"transition"` | Baseline que debe superarse en RPS medio de validación para promover |
| `log_mlflow` / `registered_model_name` | `false` / `"f1-race-position-model"` | Booleano / cadena | Registro opcional en MLflow local sin afectar la imagen batch AWS |

## Entrenamiento, calibración de ruido y reproducibilidad

- **Nivel 1 (`pace.py`):** estima $\phi$ (AR(1)), $\sigma_\varepsilon$ y los modelos de
  neumáticos **exclusivamente sobre carreras de entrenamiento** (`train_keys`), evitando
  cualquier fuga desde validación o test (verificado por regresión en
  `tests/unit/test_race_model_isolation.py`). Las carreras en mojado no aportan observación
  de ritmo seco.
- **Nivel 2 (`season.py`):** ajusta los hiperparámetros del filtro de Kalman
  ($q_\gamma, q_{\gamma,\text{season}}, q_\delta, s_c, s_w, \sigma_q, \lambda$) maximizando
  la log-verosimilitud de error de predicción solo sobre `train` tras el `burn_in_races`
  inicial. Después reproduce el filtro cronológicamente carrera a carrera con
  hiperparámetros congelados para producir el prior pre-carrera de cada Gran Premio.
- **Eventos, fricción de adelantamiento y calibración por fase (`events.py`, `forecast.py`, `runner.py`):**
  los riesgos de SC/VSC/roja y retiro (convención (a): `racedata.is_retirement`), el
  multiplicador de ruido en lluvia (`wet_noise_multiplier = 1.3915×`), las pérdidas en
  boxes, el índice de dificultad de adelantamiento en línea de meta por circuito
  (`pass_difficulty_by_location`, `8 529` ataques / `1 088` adelantamientos a $\le 1.0\text{ s}$
  sin telemetría `car_data`, tasa global `12.76%`, $\kappa = 200$), el costo por puesto
  ganado en pista (`pass_cost_seconds = 2.5 s` frente a la referencia de ritmo igual con
  solo paradas obligatorias `owed`), el plan analítico opcional de 2 paradas, la matriz de
  transición baseline y los pesos de contracción convexa LOO por fase
  (`transition_shrinkage_by_fraction`) se ajustan **exclusivamente con `train`**.
- **Reproducibilidad determinista:** el muestreador usa `np.random.default_rng(random_seed)`
  y cada artefacto registra `source_commit`, `source_dirty`, `dependency_lock_sha256`,
  `config_fingerprint` e `input_fingerprint` sobre las carreras de entrenamiento.

## Interpretabilidad estructural y diagnóstico de generalización

### Interpretabilidad estructural (`extract_interpretability`)

En equivalencia directa con la sección `interpretability` (PFI) del modelo Deep Learning,
[src/f1_race_intelligence/race_model/contracts.py](src/f1_race_intelligence/race_model/contracts.py)
genera automáticamente un bloque estructurado `interpretability` (`method:
"structural_parameter_decomposition"`, ajustado estrictamente en `train`) incluido en los
reportes de `m7-train` y `m7-evaluate` (y registrado como `interpretability.json` en MLflow
cuando se usa `--log-mlflow`):

- **Nivel 1 (`level1_pace`):** persistencia AR(1) $\phi$, ruido de vuelta limpia
  $\sigma_\varepsilon$, multiplicador de largo plazo $\sqrt{(1+\phi)/(1-\phi)}$ y tabla de
  compuestos ordenada por tasa de degradación (%/vuelta).
- **Descomposición de varianza de Nivel 2 (`level2_season_variance`):** contribución
  relativa (varianza y porcentaje) de evolución de coche en temporada ($q_\gamma^2$),
  evolución de piloto ($q_\delta^2$), ajuste equipo × circuito ($s_c^2$) y ruido transitorio
  de día de carrera ($s_w^2$), junto con el salto invernal ($q_{\gamma,\text{season}}$) y el
  peso/ruido de clasificación ($\lambda, \sigma_q$).
- **Eventos, fricción y contracción por fase (`events`, `phase_noise_multipliers`, `overtaking_friction`, `phase_transition_shrinkage`):**
  tasas globales de neutralización/retiro, pérdidas en boxes, circuitos con mayor tasa de
  SC, dificultad de adelantamiento en línea de meta (`pass_cost_seconds`, `pooled_pass_rate`,
  `wet_noise_multiplier`) y pesos de contracción LOO por fase hacia la transición.

### Diagnóstico de overfitting y underfitting (`diagnose_race_model_generalization`)

Cada corrida de `m7-evaluate --split validation` escribe `generalization_diagnosis` usando
el mismo vocabulario de 4 estados que `modeling.dl.diagnose_generalization` y consultando
**exclusivamente `train` y `validation`** (`evaluated_splits: ["train", "validation"]`; el
split `test` sellado jamás interviene):

| Estado (`status`) | Criterio exacto (`train` y `validation`) | Interpretación |
|---|---|---|
| `insufficient_metrics` | Falta el RPS de validación o el RMSE del prior | No hay métricas suficientes para emitir diagnóstico |
| `underfitting` | `validation_rps >= baseline_validation_rps` (`persist`) | El modelo no supera al baseline requerido en validación |
| `overfitting_risk` | `val_to_train_prior_rmse_ratio >= 1.25` **o** `validation_prior_z2 >= 1.50` **o** `val_to_train_rps_ratio >= 1.35` | Degradación marcada del prior, subestimación de incertidumbre ($z^2$) o brecha RPS respecto a entrenamiento |
| `balanced` | Supera al baseline y cumple todos los umbrales de generalización y calibración | Ajuste equilibrado entre entrenamiento y validación |

## Comparación experimental y evaluación real (validación 2023–2025)

El historial completo de ablaciones e iteraciones (experimentos 0–9 en Tracks A, B, C e integración final:
calibración/contracción LOO, fricción de adelantamiento sin telemetría contra referencia de
paradas obligatorias y régimen de lluvia / 2 paradas) está registrado en
[docs/m7-experiments.md](docs/m7-experiments.md).

### Resultados reportados sobre datos reales de OpenF1 (validación: 12 carreras de 2025 hasta Spa)

> **Nota de trazabilidad:** las métricas de esta subsección son **resultados reportados** de
> ejecuciones locales sobre el histórico real OpenF1 2023–2025 (excluido de git vía
> `.gitignore`). Los artefactos de esa corrida local previa (`20261122T205928432773Z`, commit
> `0e39b0b`, `source_dirty: true`, esquema de fingerprint anterior `v1`) no residen en el
> repositorio; para regenerar y verificar métricas y hashes actuales bajo
> `m7-race-model-input-fingerprint/v2` y `model.json` canónico, ejecute el protocolo de la
> sección «Instalación y comandos de ejecución».

Entrenado sobre las 46 carreras reales de 2023–2024 y evaluado sobre las 12 carreras reales
de validación de la primera mitad de 2025 (con `2025 Miami Gardens` excluida por vueltas
2–25 ausentes en origen y el conjunto `test` de 11 carreras **sellado**):

- **Parámetros estructurales ajustados en `train` (reportados):** $\phi = 0.4145$,
  $\sigma_\varepsilon = 0.4134\%$, $q_\gamma = 0.0736$, $q_{\gamma,\text{season}} = 0.2080$,
  $q_\delta = 0.0162$, $s_c = 0.2336$, $s_w = 0.2196$, $\sigma_q = 0.8775$, $\lambda = 0.9727$,
  tasa de adelantamiento en línea de meta `pass_rate = 0.1276` ($\kappa = 200$, Mónaco `2.827×`
  .. Barcelona `0.598×`), `pass_cost_seconds = 2.5 s`, `wet_noise_multiplier = 1.3915×`,
  multiplicadores de ruido por fase `{0.1: 1.0, 0.25: 2.0, 0.5: 3.0, 0.75: 3.0, 0.9: 3.0}` y
  pesos de contracción convexa LOO `{0.1: 0.1401, 0.25: 0.2194, 0.5: 0.3052, 0.75: 0.0958, 0.9: 0.3626}`.
- **Prior pre-carrera (`prior_check`, reportado):** `train` ($n = 754$) RMSE `0.4165`, $z^2 = 1.074$;
  `validation` ($n = 171$) RMSE `0.4131`, $z^2 = 0.981$ (relación val/train `0.99×`, calibración
  cercana a 1.0).
- **Diagnóstico de generalización (reportado):** `balanced` (`val/train prior RMSE = 0.99×`,
  `val z² = 0.98`, supera a `persist` y `transition`).

#### Evolución por track e integración final (Experimento 1b → 6c, 7b, 8d → 9a–9d)

| Configuración / Track | RPS Global | 10% | 25% | 50% | 75% | 90% | Pairwise LL | Δ vs `transition` [IC 95%] | Brier (`win` / `podium` / `points` / `dnf`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Referencia previa (Exp 1b) | 0.0796 | 0.1018 | 0.0980 | 0.0859 | 0.0662 | 0.0461 | 0.3667 | −0.0024 [−0.0095, +0.0049] | 0.0272 / 0.0464 / 0.1131 / 0.0390 |
| Track C — Estrategia (`8d`: lluvia + 2 paradas) | 0.0791 | 0.1030 | 0.0971 | 0.0835 | 0.0651 | 0.0466 | 0.3656 | −0.0029 [−0.0099, +0.0040] | 0.0235 / 0.0464 / 0.1129 / 0.0391 |
| Track A — Contracción LOO (`6c`) | 0.0740 | 0.0945 | 0.0924 | 0.0827 | 0.0616 | 0.0386 | 0.3458 | −0.0080 [−0.0129, −0.0031] | 0.0230 / 0.0422 / 0.1044 / 0.0390 |
| Track B — Fricción `owed-only` (`7b`) | 0.0730 | 0.0942 | 0.0907 | 0.0838 | 0.0593 | 0.0369 | 0.3389 | −0.0090 [−0.0153, −0.0022] | 0.0236 / 0.0372 / 0.1072 / 0.0390 |
| Combinado A + B (`9a`, sin Track C) | 0.0710 | 0.0925 | 0.0876 | 0.0810 | 0.0581 | 0.0357 | 0.3313 | −0.0110 [−0.0159, −0.0056] | 0.0215 / 0.0368 / 0.1028 / 0.0391 |
| Combinado A + B + C completo (`9c`, con `8a`) | 0.0709 | 0.0934 | 0.0874 | 0.0804 | 0.0574 | 0.0358 | 0.3316 | −0.0110 [−0.0161, −0.0057] | 0.0190 / 0.0368 / 0.1031 / 0.0391 |
| **Integración final A + B + C(`8b+8c`) (`9d`, `20261122T205928432773Z`)** | **0.0708** | **0.0924** | **0.0867** | **0.0810** | **0.0580** | **0.0358** | **0.3307** | **−0.0112 [−0.0161, −0.0059]** | **0.0212 / 0.0368 / 0.1025 / 0.0391** |
| Baselines: `persist` / `transition` | 0.1160 / 0.0819 | 0.1597 / 0.0996 | 0.1565 / 0.1055 | 0.1365 / 0.0965 | 0.0812 / 0.0649 | 0.0462 / 0.0432 | — / 0.4308 | — | — |

#### Desglose por fase (*k*) del modelo integrado final (`20261122T205928432773Z`)

| Fase (*k*) | `race_model` RPS | `persist` RPS | `transition` RPS | `race_model` Pairwise LL | `transition` Pairwise LL | Δ vs `persist` [IC 95%] | Δ vs `transition` [IC 95%] |
|---|---:|---:|---:|---:|---:|---|---|
| 10% | **0.0924** | 0.1597 | 0.0996 | **0.4186** | 0.4963 | −0.0673 [−0.0926, −0.0415] | **−0.0072 [−0.0137, −0.0003]** |
| 25% | **0.0867** | 0.1565 | 0.1055 | **0.4011** | 0.5258 | −0.0698 [−0.0943, −0.0445] | **−0.0188 [−0.0344, −0.0040]** |
| 50% | **0.0810** | 0.1365 | 0.0965 | **0.3784** | 0.4804 | −0.0555 [−0.0718, −0.0375] | **−0.0155 [−0.0295, −0.0031]** |
| 75% | **0.0580** | 0.0812 | 0.0649 | **0.2794** | 0.3686 | −0.0232 [−0.0346, −0.0116] | −0.0069 [−0.0144, +0.0004] |
| 90% | **0.0358** | 0.0462 | 0.0432 | **0.1760** | 0.2830 | −0.0105 [−0.0181, −0.0033] | **−0.0074 [−0.0114, −0.0044]** |
| **Global (`all`)** | **0.0708** | **0.1160** | **0.0819** | **0.3307** | **0.4308** | **−0.0452 [−0.0580, −0.0315]** | **−0.0112 [−0.0161, −0.0059]** |

- **Calibración de eventos (Brier en validación):** `win = 0.0212`, `podium = 0.0368`,
  `points = 0.1025`, `dnf = 0.0391` (`pronóstico medio 0.047 vs observado 0.041`).
- **Lectura académica honesta:** la integración final combina la fricción de adelantamiento
  escalada por dificultad de circuito frente a la referencia de solo paradas obligatorias
  (`owed-only`, Track B), la contracción convexa LOO por fase hacia `TransitionBaseline.predict_running`
  (Track A), el multiplicador de ruido en lluvia ajustado en `train` (`1.3915×`, Track C `8b`)
  y el plan analítico opcional de 2 paradas (Track C `8c`), descartando `8a` porque degradaba
  los buckets `10%` (`+0.0010`) y `25%` (`+0.0007`) al integrarse. Con ello, `race_model`
  alcanza **`0.0708` RPS global** (`−0.0088` frente a la referencia Exp 1b) y supera de forma
  **estadísticamente significativa** tanto a `persist` (`ΔRPS = −0.0452 [−0.0580, −0.0315]`)
  como a `transition` (`ΔRPS = −0.0112 [−0.0161, −0.0059]`), ganando a ambos baselines en **los 5
  buckets de distancia (10% a 90%, con IC 95% estrictamente negativo en 4/5 buckets individuales)**
  y en **11 de las 12 carreras de validación** (incluyendo Mónaco `0.0896 → 0.0701`,
  Montréal `0.1075 → 0.0908`, Silverstone `0.0904 → 0.0750`, Spa `0.0733 → 0.0577`,
  Melbourne `0.1015 → 0.0971` y Barcelona `0.1212 → 0.1111`).

## Anti-leakage y contrato unificado M7 → Registro / MLflow → Publicación → Serving

### Guardas anti-leakage

1. **Corte estricto en la vuelta *k* (`racedata.truncate_at_lap`):** elimina vueltas,
   paradas, mensajes de control de carrera e intervalos posteriores al cruce del líder en la
   vuelta *k*, recorta stints y borra el resultado oficial. `forecast.car_states` rechaza
   cualquier carrera sin truncar o que conserve `result`.
2. **Aislamiento de parámetros (`tests/unit/test_race_model_isolation.py`):** reemplazar
   las carreras de validación (orden invertido y ruido extremo) no altera ningún parámetro
   ajustado ($\phi$, $\sigma_\varepsilon$, neumáticos, hiperparámetros del filtro, eventos,
   distancias, transición, multiplicadores de ruido ni el `input_fingerprint` de
   entrenamiento).
3. **Conjunto `test` sellado (`race_model.test_sealed`):** ninguna etapa abre archivos raw
   de `test`. La única entrada permitida es `m7-evaluate --split test --unseal-test`, que
   escribe primero en `test_ledger.json`, prohíbe promover y rechaza una segunda mirada sin
   `--allow-repeat-test`.

### Empaquetado inmutable y verificación SHA-256

Al guardar cada versión inmutable mediante `ModelRegistry.save`, el modelo serializa tanto
`model.json` como `manifest.json` en **JSON canónico UTF-8** (claves ordenadas, sin espacios
superfluos, sin `NaN`) y publica mediante `publish_version_package` con escritura condicional
(`write_bytes_if_absent`), verificación previa al commit y marcador `PUBLISHED` escrito al
final, en paridad con [M07_MODEL_PUBLISHING.md](M07_MODEL_PUBLISHING.md):

```text
data/models/m7/version=<versión>/          (o models/m7/version=<versión>/ en layout lake)
├── model.json                             artefacto completo en JSON canónico UTF-8
├── manifest.json                          JSON canónico UTF-8 con SHA-256 y tamaño de cada archivo
└── PUBLISHED                              sha256:<hash de manifest.json> (escrito al final)
```

- **Atomicidad y reintentos idempotentes:** un fallo parcial antes de escribir `PUBLISHED`
  deja la versión invisible para `versions()`, bloqueada para `load()` y `promote()`, y
  recuperable al reintentar `save()` con el mismo artefacto. Un reintento idéntico tras la
  publicación retorna la URI verificada; cualquier intento de sobrescribir una versión con
  contenido distinto falla con `FileExistsError`.
- **Verificación obligatoria antes de usar o promover:** tanto `ModelRegistry.load()` como
  `ModelRegistry.promote()` y `ModelRegistry.read_model_bytes()` ejecutan verificación completa
  de `PUBLISHED`, `manifest.json` canónico, conjunto exacto de objetos (sin archivos extra) y
  SHA-256/tamaño de `model.json`, rechazando paquetes incompletos o manipulados con
  `ModelPackageIntegrityError`.
- **`--stage verify-m7`:** homólogo de `--stage verify-m6` para verificación de solo lectura.

### Reportes generados y MLflow opcional (perezoso y predictivo)

- **Reportes en almacenamiento:** cada `m7-evaluate` genera tanto el respaldo estructurado
  `evaluate_<split>.json` (con `provenance` e `input_fingerprint` `v2` sobre todas las
  observaciones de `train`, `generalization_diagnosis`, `interpretability`,
  `package_verification` y métricas) como el informe académico consolidado
  `evaluate_<split>.md` en 5 secciones paralelas a `mlp_comparison_<id>.md`.
- **MLflow local opcional (`--log-mlflow`):** cuando está instalado el extra `training`,
  `log_race_model_to_mlflow` importa perezosamente `mlflow` y el módulo dedicado
  `race_model/mlflow_pyfunc.py` (`RaceModelPyfuncModel`), sin cargar MLflow al importar los
  módulos base del modelo. Registra la corrida en la misma base SQLite
  (`data/mlflow/tracking.db`) y experimento (`f1-race-intelligence`) que `LinearRegression` y
  `TabularMLP`, y su método `predict()` ejecuta el contrato predictivo real (`LoadedModel.run`
  + `validate_forecast_response`) con paridad verificable contra `RaceModelRunner.forecast`.

### Contrato de inferencia / serving (`contracts.py` y `lambda_handler.py`)

En paridad con `serving/contract.py` (M8), toda petición a `lambda_handler.handler` pasa
por `validate_forecast_request` (rechaza claves desconocidas, tipos inválidos, combinaciones
inconsistentes en modo `live`, y `lap > total_laps`) y toda salida de `RaceModelRunner.forecast`
(y de `RaceModelPyfuncModel.predict`) pasa por `validate_forecast_response` (exige campos
obligatorios y rangos válidos de `lap`/`total_laps`, pilotos únicos y consistentes con la
matriz, estructura 2D $N \times N$ sin valores fuera de $[0, 1]$, suma de filas y columnas
igual a `1.0`, `p_dnf` estrictamente en $[0, 1]$ y consistencia con `p_position`/`ahead`).

## Aislamiento de dependencias y portabilidad a AWS

- **Runtime base limpio:** `race_model` depende únicamente del runtime base
  ([requirements/runtime-lock.txt](requirements/runtime-lock.txt), que ya incluye `scipy`),
  sin `mlflow` ni `torch`. Toda referencia a MLflow vive en `race_model/mlflow_pyfunc.py` y se
  importa únicamente de forma perezosa dentro de `log_race_model_to_mlflow()` (`log_mlflow: false`
  por defecto), garantizando `"mlflow" not in sys.modules` al importar los módulos base incluso
  cuando MLflow está instalado.
- **Portabilidad AWS ya definida en IaC:** a diferencia del prototipo local de M8, el modelo
  de carrera ya cuenta con infraestructura SAM en [infra/template.yaml](infra/template.yaml),
  state machine [infra/statemachine/model.asl.json](infra/statemachine/model.asl.json)
  (entrenamiento/evaluación/promoción y actualización semanal en ECS Fargate) y contenedor
  Lambda de pronóstico ([Dockerfile.forecast](Dockerfile.forecast)), manteniendo los
  calendarios desactivados por defecto (ver [infra/README.md](infra/README.md#modelo-de-carrera-m7-desplegar-y-mantener)).
- **Bucket S3 destino:** los datos se cargarán desde
  `s3://f1-race-intelligence-dev-793264560810-us-east-1` vía `F1_STORAGE_BUCKET` (layout
  `lake`, sin fijarlo en YAML ni versionar credenciales; los datos aún no están cargados en
  ese bucket — ver [M07_MODEL_COMPARISON.md](M07_MODEL_COMPARISON.md#5-procedencia-de-datos-y-destino-de-carga-en-aws-s3)).

## Instalación y comandos de ejecución

```bash
# Runtime base (entrenamiento, evaluación, verificación y pronóstico sin MLflow)
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -e ".[dev]" --no-deps

# Opcional: añadir MLflow para registrar corridas junto a LinearRegression y TabularMLP
python -m pip install -r requirements/training-lock.txt
python -m pip install -e ".[dev,training]" --no-deps
```

```bash
# 1. Entrenar, evaluar en validación y promover si supera al baseline 'persist'
python -m f1_race_intelligence --stage m7-train,m7-evaluate --promote

# 2. Igual, registrando además métricas, informe .md y artefacto pyfunc en MLflow local
python -m f1_race_intelligence --stage m7-train,m7-evaluate --promote --log-mlflow

# 3. Verificar integridad SHA-256 (PUBLISHED + manifest.json + model.json)
python -m f1_race_intelligence --stage verify-m7

# 4. Pronóstico en carrera desde la vuelta k (raw o en vivo)
python -m f1_race_intelligence --stage m7-forecast --year 2025 --session-key 9998 --lap 30

# 5. Actualizar estado de temporada tras un nuevo GP (hiperparámetros congelados)
python -m f1_race_intelligence --stage m7-update

# 6. Inspeccionar en MLflow junto a los modelos tabulares
mlflow ui --backend-store-uri sqlite:///data/mlflow/tracking.db
```

## Protocolo de cierre con datos reales (test sellado)

El entrenamiento y la evaluación en validación con datos reales de 2023–2025 (primera mitad)
ya están completados y verificados. El conjunto `test` (11 carreras: Budapest a Abu Dabi
2025) permanece **sellado** (`test_sealed: true`). Para el informe final de sustentación,
una única ejecución registrada en `test_ledger.json` desella y evalúa el test sin promover:

```bash
python -m f1_race_intelligence --stage m7-evaluate --split test --unseal-test
```

## Validación, pruebas y honestidad académica

```bash
# Pruebas del modelo de carrera: matemática, anti-leakage, test sellado, Lambda y contratos M7
python -m pytest \
  tests/unit/test_race_model.py \
  tests/unit/test_race_model_isolation.py \
  tests/unit/test_race_model_lambda.py \
  tests/unit/test_race_model_lines.py \
  tests/unit/test_race_model_contracts.py

# Suite completa no integrada
python -m pytest -m "not integration"
```

**Declaración de honestidad académica:**

- Las pruebas automatizadas en `tests/unit/` usan carreras sintéticas generadas en memoria
  y sin acceso a red; prueban la corrección técnica y las guardas anti-leakage, no el
  desempeño real.
- Las métricas de la sección «Resultados medidos sobre datos reales de OpenF1» provienen
  exclusivamente de la evaluación controlada sobre las 12 carreras reales de **validación**
  de 2025 (entrenado en 2023–2024); el conjunto `test` (segunda mitad de 2025) no ha sido
  abierto ni usado en ninguna decisión de modelado.
- La infraestructura AWS definida en `infra/` está validada estáticamente y en pruebas
  unitarias, no desplegada en producción real para M7 (ver [infra/aws_status.json](infra/aws_status.json)).

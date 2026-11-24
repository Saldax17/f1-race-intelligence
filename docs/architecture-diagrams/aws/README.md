# Diagrama de arquitectura AWS — F1 Race Intelligence

Reconstruido el 2026-11-22 a partir **exclusivamente** del repositorio en
`HEAD efd0a9c05e2ee12d3162e83a5dd5d25e26245fa5` (rama `docs/record-meeting-key-deployment`,
árbol limpio). No se ejecutó ningún comando de AWS ni se tocó código, infraestructura o
configuración existentes.

| Archivo | Uso |
|---|---|
| `f1-race-intelligence-aws-architecture_2026-11-22.drawio` | Editable en draw.io (iconos embebidos, texto y flechas editables) |
| `f1-race-intelligence-aws-architecture_2026-11-22.svg` | Vectorial, 16:9 (`viewBox 1920×1080`), escala sin pérdida |
| `f1-race-intelligence-aws-architecture_2026-11-22.png` | 1920×1080 para PowerPoint/PDF |
| `source/build_diagram.py` + `source/icons/` | Generador único del `.drawio` y el `.svg` (`python -I build_diagram.py <salida> <nombre>`) |

No se sobrescribió ninguna imagen previa: es una carpeta nueva con nombre fechado.

**Revisión de estado (2026-11-22, posterior a la primera versión).** Se regeneraron el `.drawio`, el
`.svg` y el `.png` solo para alinear los textos de estado con `infra/aws_status.json` e
`infra/README.md` (runtime `800931b`, task definition revisión 2, M7 ausente del stack). El
diseño y la disposición no cambian. Tampoco esta revisión consultó AWS.

## Cómo leer el diagrama

| Marca | Significado |
|---|---|
| **A** borde azul sólido | Definido en IaC (`infra/template.yaml`) |
| **C** círculo ámbar «?» | Despliegue en AWS **reportado, no reproducible desde el repositorio** (stack `dev`, plantilla anterior a M7). Lo llevan los recursos AWS de ese stack |
| **B** borde verde | Validado en local / simulación, sin AWS |
| **D** gris punteado | Futuro / pendiente, no implementado |
| Flechas ámbar discontinuas | Compuerta cerrada: `PipelineEnabled=false` |

`infra/aws_status.json` declara `deployment` = **DESPLEGADO PARCIALMENTE** (reportado, sin M7) y
`real_environment_test` = **NO VERIFICABLE**. El «?» no dice que el recurso falte, sino que el
repositorio no puede demostrar que existe. Los recursos de M7 se dibujan como definidos en IaC
(`master`) y ausentes del stack desplegado.

## Flujo arquitectónico

1. **Disparo.** Una regla programada de EventBridge (`rate(15 minutes)`, `State=DISABLED` por
   defecto) invoca la Lambda `availability`, que sondea OpenF1
   (`GET /v1/sessions?session_key=7953`) y devuelve `{available, reason}`. Un OpenF1 no
   disponible es una respuesta normal, no un error.
2. **Orquestación.** El resultado (destino OnSuccess) llega al bus `…-orchestration`; la regla
   `availability-result` invoca `pipeline-admission`, que reserva en S3
   (`manifests/control/pipeline-cadence.json`, escritura condicional) un máximo de un inicio
   automático por `PipelineIntervalHours` y llama a `StartExecution`.
3. **State machine** (`pipeline.asl.json`): valida el resultado → `IsOpenF1Available` →
   **`IsPipelineEnabled`** (con `false`, valor por defecto, termina en `PipelineDisabled` sin
   ejecutar nada) → `CheckForConcurrentRun` → `ResolveExtractionScope`.
4. **Procesamiento.** Si estuviera habilitado: `PlanSessions` (tarea Fargate `--stage plan`, escribe
   `manifests/plan/<ejecución>.json`) → `LoadPlan` → **Map por sesión, `MaxConcurrency=1`**
   (M2 → M3 → M4 → M5, una tarea Fargate por etapa vía `ecs:runTask.sync`) → M6 global
   (`--require-complete-history`; exit 3 = diferido). Cada tarea valida `Containers[0].ExitCode`;
   los fallos van a `NotifyPipelineFailure` (SNS). Todas usan una misma task definition y la imagen de
   ECR fijada por digest, en una VPC dedicada con subredes públicas, sin ingress y con egress 443.
5. **Almacenamiento.** Cada etapa lee/escribe solo su zona del lake: M2 → `raw/`, M3 → `validated/`,
   M4 → `consolidated/`, M5 → `features/`, M6 → `modeling/`; los planes y reportes en `manifests/`.
   `models/` y `forecasts/` pertenecen a M7.
6. **Observabilidad y alertas.** Log groups de Lambdas, state machine y tareas; 6 alarmas base
   (+2 de M7) y un dashboard; las alarmas y los `OnFailure` publican en el topic SNS, con
   e-mail opcional (`AlertEmail`).
7. **M7/M8.** El modelo de carrera M7 (Fargate + Step Functions propia + Lambda de pronóstico) está
   **definido en `master`** pero, según `infra/README.md`, no está en el stack desplegado, y sus
   calendarios están `DISABLED`. MLflow (SQLite) y el serving M8 (FastAPI) existen **solo en local**;
   su despliegue en AWS figura como pendiente.

## Trazabilidad: componente → evidencia en el repositorio

Estado: **A** = definido en IaC · **B** = validado localmente · **C** = despliegue en `dev` reportado, no reproducible desde el repositorio ·
**D** = futuro. «Evidencia de validación» son pruebas/simulaciones locales, no AWS.

| Componente del diagrama | Archivo / recurso que lo demuestra | Estado |
|---|---|---|
| OpenF1 API (externa) | `OpenF1BaseUrl` en `infra/template.yaml`; `infra/functions/availability_checker/app.py` | Externa (no AWS) |
| Regla programada EventBridge | `AvailabilityScheduleRule`, parámetros `AvailabilitySchedule/State` en `infra/template.yaml` | A · C |
| Lambda `availability` | `AvailabilityCheckerFunction` + `infra/functions/availability_checker/app.py`; `tests/infra/test_availability_checker.py` | A · B · C |
| Bus `orquestación` + regla `availability-result` | `OrchestrationEventBus`, `AvailabilityResultRule` en `infra/template.yaml` | A · C |
| Lambda `pipeline-admission` | `PipelineAdmissionFunction`, `infra/functions/pipeline_admission/handler.py`; `tests/infra/test_pipeline_admission.py` (Fake S3) | A · B · C |
| Step Functions `…-pipeline` | `PipelineStateMachine`, `infra/statemachine/pipeline.asl.json`; `tests/infra/test_state_machine.py` | A · B · C — definición con `--meeting-key` y referencia a la task definition revisión 2 (reportado); no se ha ejecutado |
| Compuerta `PipelineEnabled=false` | Parámetro `PipelineEnabled` + estado `IsPipelineEnabled` en `pipeline.asl.json`; `infra/samconfig.toml` | A · B |
| Tareas plan/M2–M6 (`runTask.sync`, Map `MaxConcurrency=1`) | Estados `PlanSessions`, `PerSessionPipeline`, `M6_ModelingDataset` en `pipeline.asl.json` | A · B · C |
| Clúster ECS + task definition Fargate | `PipelineCluster`, `PipelineTaskDefinition` en `infra/template.yaml` (condicional a `PipelineImageUri`) | A · C — revisión 2 `ACTIVE` con la imagen `800931b` (reportado); el smoke del 2026-11-21 usó la revisión 1 |
| VPC, subredes, Internet Gateway, security group | `PipelineVpc`, `PipelineSubnetA/B`, `PipelineInternetGateway`, `PipelineTaskSecurityGroup` | A · C |
| ECR `…-pipeline` | `PipelineImageRepository`; `Dockerfile`; `tests/infra/test_container_image.py` | A · B · C — imagen `800931b` publicada (reportado) |
| S3 data lake y sus zonas | `DataBucket`, `DataBucketPolicy`; `src/f1_race_intelligence/storage/layout.py` (`LAKE_ROOTS`); `tests/unit/test_storage_layout.py` | A · B · C |
| IAM (10 roles en `master`) | Roles `*Role` en `infra/template.yaml`; `infra/README.md` § IAM; `tests/infra/test_template.py` | A · B · C para los 5 roles de la fila superior del diagrama (reportados como desplegados y sin cambios); los 5 de M7 (trazo discontinuo) son A (master), sin desplegar |
| Log groups, alarmas, dashboard | `*LogGroup`, `*Alarm`, `ObservabilityDashboard` en `infra/template.yaml` | A · C |
| SNS `critical-alerts` | `AlertTopic`, `AlertTopicPolicy`, `AlertEmailSubscription` | A · C |
| M7 race-model (SFN + Fargate + schedules) | `RaceModel*` en `infra/template.yaml`, `infra/statemachine/model.asl.json`; `tests/unit/test_race_model*.py` | A (master) — **ausente** del stack desplegado, según `infra/README.md` («Pendientes», punto 2); no se marca C |
| Lambda `race-forecast` + ECR forecast | `ForecastFunction`, `ForecastImageRepository`, `Dockerfile.forecast`; `tests/unit/test_race_model_lambda.py` | A (master) — ausente del stack desplegado; requiere `ForecastImageUri` |
| M7 + MLflow (SQLite) local | `M07_TRAINING_MLFLOW.md`; `tests/unit/test_training.py` | B |
| Publicación del paquete de modelo | `M07_MODEL_PUBLISHING.md` | B |
| M8 serving FastAPI | `src/f1_race_intelligence/serving/app.py`; `tests/unit/test_serving_api.py`, `test_m7_m8_integration.py`; `DEMO_LOCAL.md` | B |
| Serving M8 en AWS, MLflow remoto, dashboard de negocio, drift, CD | `README.md` §19 («Próximos pasos») | D |

## Puntos que no se pueden verificar desde el repositorio

- **Despliegue real.** `infra/aws_status.json` → `deployment` = DESPLEGADO PARCIALMENTE, `real_environment_test`
  = NO VERIFICABLE. `infra/README.md` (secciones «Validación de runtime AWS» del 2026-11-21, «Despliegue
  de la corrección de `meeting_key`» del 2026-11-21 y «Despliegue de la imagen `800931b`» del 2026-11-22)
  describe un stack `dev` en `us-east-1` *medido con la AWS CLI*. Eso es un **reporte**: no se verificó
  de forma independiente desde AWS, no se ejecutó la state machine completa y el smoke sintético
  (`copy-raw` y M3–M5) usó la task definition revisión 1 y la imagen `3d3b4e6`, no la vigente. El
  diagrama no da por demostrado lo que solo está *reportado*. No hay evidencia suficiente para
  afirmar un despliegue productivo; `PipelineEnabled=false` y los calendarios siguen `DISABLED`.
- **M7 en AWS.** La plantilla de `master` lo define; el stack desplegado (44 recursos, reportado) es
  anterior y no lo incluye (aunque la imagen `800931b` ya contiene el código de M7). Cuántos recursos
  crearía un despliegue sale de leer condiciones, no de un change set.
- **Región.** `us-east-1` sale de `infra/samconfig.toml` (entorno `dev`); no se comprobó contra AWS.
- **ROADMAP externo.** `docs/ROADMAP.md` está parcialmente desactualizado (su propio encabezado lo
  indica, revisado sobre `39ee675`); no se usó como fuente.
- **Render del `.drawio`.** Se validó su estructura (XML, ids únicos, 27 aristas con origen y destino
  válidos, 19 iconos decodificables), pero no se abrió en draw.io desde aquí; el SVG y el PNG sí se
  renderizaron con Chrome y se revisaron visualmente.

## Iconos

SVG de la serie *AWS Architecture Icons* (`Arch_*_48` / `Res_*_48`, según el título interno de cada
archivo), tomados del paquete npm `aws-icons@3.3.0` (redistribución de terceros, licencia MIT del
empaquetado). **No se descargó el paquete oficial desde aws.amazon.com**; antes de publicar fuera del
ámbito académico conviene contrastarlos con la versión vigente y sus términos de uso. Los iconos
se incrustan en el `.drawio` como imágenes (no como formas nativas `mxgraph.aws4`).

## Notas de estilo

Fondo oscuro. Los acentos reutilizan la paleta de las figuras existentes del proyecto
(`openf1_fuentes_de_datos.png`: violeta para el pipeline, verde azulado para datos, salmón para
OpenF1), pero esas figuras son claras y la presentación completa no estaba en el repositorio, así que la
coherencia es aproximada.

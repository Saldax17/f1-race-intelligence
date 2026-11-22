# Infraestructura AWS (SAM / CloudFormation)

> **Estado:** `aws_status.json` es la fuente única de verdad versionada. La
> arquitectura y su código están definidos, y la validación local está
> registrada allí. El estado de despliegue y las pruebas en AWS real son
> **NO VERIFICABLES** en ausencia de comprobación humana; este documento no
> reemplaza esa evidencia. No entrena ni publica modelos (sin M7, sin SageMaker).

## Estructura

```
infra/
├── aws_status.json                       # estado técnico AWS: fuente única de verdad
├── template.yaml                          # stack SAM (CloudFormation + Transform)
├── samconfig.toml                         # parámetros por entorno (dev, prod); confirma changesets
├── functions/
│   ├── availability_checker/
│   │   ├── app.py                         # Lambda: sondeo liviano a OpenF1 (solo stdlib)
│   │   └── requirements.txt               # vacío a propósito
│   └── pipeline_admission/
│       ├── handler.py                     # Lambda: cadencia automática persistida en S3
│       └── requirements.txt               # cierre transitivo fijado para escrituras condicionales
└── statemachine/
    └── pipeline.asl.json                  # Step Functions STANDARD: disponibilidad → M2…M6
tests/infra/                               # tests de la Lambda, la state machine y el template
```

## Diagrama lógico

```
                    EventBridge rule (schedule, default rate(15 minutes))
                                     │ invoke (async)
                                     ▼
                  Lambda  openf1-availability  ──────── GET /v1/sessions?session_key=7953 ──► OpenF1
                     │                     │
         OnSuccess   │                     │ OnFailure (crash/timeout de la función)
     (JSON resultado)▼                     ▼
      EventBridge bus  <p>-<env>-orchestration      SNS critical-alerts ──► e-mail (opcional)
                     │                                 ▲        ▲
      rule availability-result                         │        │
      InputPath $.detail.responsePayload               │   CloudWatch alarms
                     ▼                                 │   · errores de las Lambdas
      Lambda pipeline-admission ─ OnFailure (tras reintentos) ─► SNS critical-alerts
      CAS persistente en S3 (manifests/control/)       ▲
                     │ inicia como máximo 1 pipeline por intervalo
                     ▼
      Step Functions STANDARD  <p>-<env>-pipeline      │   · ejecuciones fallidas/timeout/abortadas
        LoadContext                                    │   · FailedInvocations de las reglas
        IsAvailabilityResultValid ── no ─► Fail        │   · OpenF1 no disponible 6 h seguidas (EMF)
        IsOpenF1Available ── false ─► OpenF1Unavailable (Succeed)
        IsPipelineEnabled ── false ─► PipelineDisabled (Succeed)   ← valor por defecto
        CheckForConcurrentRun ── otra ejecución RUNNING ─► AnotherRunInProgress (Succeed)
        PlanSessions        Fargate --stage plan --plan-output s3://…/manifests/plan/<ejecución>.json
        LoadPlan            s3:getObject del plan → $.plan.document.sessions
        PerSessionPipeline  Map, concurrencia 1, una tarea Fargate por etapa
          M2_Extraction → M3_Validation → M4_Consolidation → M5_Features
        M6_ModelingDataset  Fargate --stage m6 (global)
        cada tarea: Retry (arranque / ejecución) + Choice sobre el exit code
        Catch ─► NotifyPipelineFailure ─┘ ─► PipelineFailed

  S3 data lake <p>-<env>-<account>-<region>      ECR <p>-<env>-pipeline      CloudWatch Logs (retención parametrizada)
  raw/ validated/ consolidated/ features/        (imagen batch M2–M6,        /aws/lambda/…  /aws/vendedlogs/states/…
  modeling/ models/ manifests/                    fijada por digest)          /aws/ecs/<p>-<env>-pipeline (tareas)

  VPC 10.42.0.0/24 (dedicada): 2 subredes públicas, Internet Gateway, sin NAT ni endpoints
  ECS cluster <p>-<env>-pipeline (sin Container Insights) · task definition Fargate x86_64 0,5 vCPU / 1 GB
```

El availability checker solo sondea OpenF1 a la frecuencia de
`AvailabilitySchedule`; su resultado viaja por el *destination* OnSuccess a un
bus de EventBridge. La regla invoca `pipeline-admission`, que reserva de forma
atómica en S3 el intervalo automático antes de iniciar Step Functions. Los
sondeos continúan aunque el pipeline esté limitado a una ejecución cada
`PipelineIntervalHours`; un sondeo no consume por sí solo el intervalo.

## Recursos

| Recurso | Configuración |
|---|---|
| S3 `DataBucket` | versionado; SSE-S3; bloqueo total de acceso público; `BucketOwnerEnforced`; política que niega tráfico sin TLS; lifecycle: aborta multipart a 7 d, expira versiones no actuales (parámetro, 90 d), limpia delete markers, `raw/` a Intelligent-Tiering a 30 d, `manifests/` a 90 d, `logs/` expira a 365 d; **nunca expira datasets actuales**; `DeletionPolicy: RetainExceptOnCreate`, `UpdateReplacePolicy: Retain` |
| Lambda `openf1-availability` | Python 3.12 arm64, 128 MB, 15 s; sin VPC; sin reintentos asíncronos; destinos OnSuccess→bus, OnFailure→SNS; métricas EMF `OpenF1Available`, `OpenF1LatencyMs` |
| Lambda `pipeline-admission` | Python 3.12 arm64, 128 MB, 15 s; invocada por la regla de resultados; reserva el intervalo automático en S3 con `If-None-Match`/`If-Match` y solo entonces llama `StartExecution` |
| EventBridge | regla programada (parámetros `AvailabilitySchedule`, `AvailabilityScheduleState`); bus propio + regla de resultados |
| Step Functions | STANDARD; logs `ALL` a CloudWatch; timeout global 12 h; plan y M2–M6 como `ecs:runTask.sync` (ver "Tareas Fargate") |
| ECS | cluster sin Container Insights; task definition Fargate `X86_64`, 512 CPU / 1024 MB, usuario `10001:10001`, raíz de solo lectura, `capabilities.drop: [ALL]`, sin puertos ni volúmenes; solo si `PipelineImageUri` tiene valor |
| Red | VPC dedicada `10.42.0.0/24`, 2 subredes públicas en 2 AZ, Internet Gateway, security group sin entrada y salida solo TCP 443 |
| SNS | topic de alertas críticas; política para CloudWatch de la misma cuenta; suscripción e-mail opcional |
| ECR | tags inmutables, escaneo al subir, expira sin tag a 7 d, conserva 30 imágenes; `RetainExceptOnCreate` |
| CloudWatch | 4 log groups con retención `LogRetentionDays`; 5 alarmas base y una condicional; dashboard operativo |

## IAM (mínimo privilegio)

| Rol | Confía en | Permisos |
|---|---|---|
| `…-availability-lambda` | `lambda.amazonaws.com` | escribir en su log group; `events:PutEvents` en el bus de orquestación; `sns:Publish` en el topic de alertas. **Sin S3.** |
| `…-pipeline-states` | `states.amazonaws.com` | `sns:Publish` en el topic; APIs de entrega de logs (`Resource: "*"` exigido por AWS); `ecs:RunTask` solo de la familia `<p>-<env>-pipeline` y con condición `ecs:cluster`; `ecs:DescribeTasks`/`StopTask` solo sobre tareas del cluster; `events:PutRule/PutTargets/DescribeRule` solo sobre la regla administrada `StepFunctionsGetEventsForECSTaskRule`; `iam:PassRole` solo de los dos roles de tarea y con `iam:PassedToService=ecs-tasks.amazonaws.com`; `s3:GetObject` solo en `manifests/plan/*`; `states:ListExecutions` solo sobre esta state machine |
| `…-pipeline-admission` | `lambda.amazonaws.com` | escribir en su log group; `s3:ListBucket` condicionado exactamente a `manifests/control/pipeline-cadence.json`; `GetObject`/`PutObject` solo en ese objeto; `states:StartExecution` solo en esta state machine; `sns:Publish` solo en el topic |
| `…-data-processing` (tarea) | `ecs-tasks.amazonaws.com` | `s3:ListBucket` del bucket; `GetObject`/`PutObject` solo en `raw/ validated/ consolidated/ features/ modeling/ manifests/{extraction,incremental,consolidation,features,modeling,plan}/`. No puede escribir `manifests/control/`. Sin `Delete*`, sin `models/`. |
| `…-data-processing-exec` (ejecución) | `ecs-tasks.amazonaws.com` | pull de la imagen del repositorio ECR (`GetAuthorizationToken` requiere `*`); escribir en `/aws/ecs/…` |

Las relaciones de confianza de Step Functions, EventBridge y ECS llevan
`aws:SourceAccount`. Los dos roles Lambda usan la confianza estándar de
`lambda.amazonaws.com`: no se pudo verificar que Lambda admita esa condición en
roles de ejecución, y si no la admite `CreateFunction` podría fallar. No hay
políticas administradas de AWS ni `AdministratorAccess`.

`ecs:DescribeTasks`/`StopTask` se limitan a `task/<cluster>/*` aunque la
política que genera AWS usa `"*"`; la condición `ecs:cluster` y ese ámbito se
verifican en la primera ejecución controlada.

`PipelineAdmissionRole` limita `s3:ListBucket` con `StringEquals` al prefijo
exacto del objeto de cadencia, sin dar acceso de listado general. La Lambda no
interpreta el error de `GetObject` para decidir si la reserva existe (sin
`ListBucket` efectivo S3 responde 403 también para una clave ausente): llama a
`ListObjectsV2` con `Prefix` igual a la clave y exige que un resultado tenga
`Key` idéntica (una clave que solo comparte el prefijo no cuenta). Si existe, la
lee con `GetObject`; si no, la crea con `PutObject` y `If-None-Match: *`. Un 403
del listado se propaga y falla de forma cerrada. Con un 409/412 relee y resuelve
con el CAS. Esta semántica está modelada en un Fake S3; el comportamiento real
de IAM/S3 sigue sin verificarse en AWS. El rol Fargate conserva `ListBucket` del bucket porque sus chequeos `HeadObject`
necesitan distinguir objetos ausentes; sus recursos de escritura excluyen
explícitamente `manifests/control/`, por lo que no puede cambiar la reserva.

## Parámetros

| Parámetro | Por defecto | Nota |
|---|---|---|
| `ProjectName` | `f1-race-intelligence` | prefijo de nombres y tag |
| `Environment` | `dev` | `dev` · `staging` · `prod` |
| `S3BucketName` | vacío | vacío ⇒ `<project>-<env>-<account>-<region>` |
| `AvailabilitySchedule` | `rate(15 minutes)` | solo valores ≥ 15 min; la alarma sostenida solo se crea para 15/30 min |
| `AvailabilityScheduleState` | `DISABLED` | `samconfig.toml` también lo fija en `DISABLED` (dev y prod); activarlo es una decisión explícita |
| `PipelineIntervalHours` | `24` | intervalo mínimo entre inicios automáticos tras una comprobación positiva; mínimo permitido 24 horas. No limita inicios manuales autorizados |
| `PipelineEnabled` | `false` | `true` exige `PipelineImageUri` (regla `PipelineNeedsAnImage`); dejar en `false` hasta verificar una primera ejecución controlada |
| `PipelineImageUri` | vacío | imagen batch **por digest** (`…/f1-race-intelligence-dev-pipeline@sha256:<64 hex>`); vacío ⇒ no se crea la task definition. No se guarda en `samconfig.toml` |
| `LogRetentionDays` | `30` | valores válidos de CloudWatch |
| `NoncurrentVersionExpirationDays` | `90` | |
| `OpenF1BaseUrl` | `https://api.openf1.org/v1` | |
| `AlertEmail` | vacío | vacío ⇒ **sin suscripción: nadie recibe las alertas**; ver "Alertas por correo" |

## Observabilidad operativa

Las tareas Fargate reciben `F1_PIPELINE_EXECUTION_ID` desde Step Functions. El
formateador JSON añade `component`, `logger` y `pipeline_execution_id` a cada
registro. `component` usa el vocabulario estable configurado por `F1_COMPONENT`
(por defecto `f1-race-intelligence`; `pipeline` en ECS y
`availability-checker` en Lambda); `logger` conserva el nombre Python real del
módulo. El CLI ya registra `stage_started` y `stage_finished` con duración, estado,
resumen y excepciones. En M3, `WARNING` se etiqueta `partial` y `FAIL` se
registra como `failed` (y se muestra como fallo en modo no estricto); ninguno
cambia el código de salida no estricto ni detiene la secuencia. Un M6
`VALID_WITH_WARNINGS` se etiqueta como `partial`. El resumen M6 conserva `run_id` e
`input_fingerprint`; así se puede enlazar ese artefacto con el ID de ejecución
del pipeline sin cambiar el contrato del reporte M6. La Lambda registra el
componente, el estado `completed`, hora de inicio/fin y duración. Un sondeo
HTTP no disponible sigue siendo una invocación completada: no es un error de
ejecución Lambda.

Los logs se escriben en los grupos descritos arriba y retienen el periodo
configurado. `pipeline-admission` emite líneas JSON con
`component=pipeline-admission`, `status`, `check_id`, `correlation_id` derivado
por hash y `execution_name` cuando se conoce. No se registra el evento completo,
ni secretos, credenciales o contenido de artefactos.
Los IDs de ejecución, `run_id`, `input_fingerprint`, estados y rutas de
artefactos son metadatos de diagnóstico, no sus contenidos. La ejecución M7 /
MLflow y el prototipo M8 no forman parte de esta infraestructura desplegable;
no hay métricas AWS de entrenamiento, publicación ni HTTP 5xx del serving.

| Métrica | Namespace / dimensión | Unidad y estadística | Significado |
|---|---|---|---|
| `ExecutionsStarted`, `ExecutionsSucceeded`, `ExecutionsFailed`, `ExecutionsTimedOut`, `ExecutionsAborted` | `AWS/States`, `StateMachineArn=<pipeline>` | Count / Sum | Volumen y desenlace de ejecuciones |
| `ExecutionTime` | `AWS/States`, `StateMachineArn=<pipeline>` | Milliseconds / Average, Maximum | Duración de las ejecuciones |
| `Invocations`, `Errors`, `Throttles` | `AWS/Lambda`, `FunctionName=<availability checker>` y `<pipeline admission>` | Count / Sum | Actividad, errores de ejecución y limitación de concurrencia |
| `Duration` | `AWS/Lambda`, `FunctionName=<availability checker>` y `<pipeline admission>` | Milliseconds / Average, Maximum | Duración de cada Lambda |
| `FailedInvocations` | `AWS/Events`, `RuleName`; para la regla de resultados también `EventBusName` | Count / Sum | Fallos al invocar la Lambda o iniciar Step Functions |
| `OpenF1Available` | `f1-race-intelligence/<env>`, `Environment=<env>` | Count / Maximum (0 o 1) | Resultado del sondeo, emitido por EMF |
| `OpenF1LatencyMs` | `f1-race-intelligence/<env>`, `Environment=<env>` | Milliseconds / Average | Latencia del sondeo, emitida por EMF |

`<env>` y el prefijo del namespace se generan desde `ProjectName` y
`Environment`. Las métricas `AWS/*` son métricas de servicio; las dos
`OpenF1*` son las únicas métricas personalizadas. La Lambda emite EMF a su log
group y no necesita `cloudwatch:PutMetricData`.

### Alarmas y dashboard

Las cinco alarmas base y, cuando la cadencia lo permite, la alarma sostenida
publican al mismo `AlertTopic`; las acciones `OKActions` también notifican la
recuperación:

| Alarma | Criterio | Datos faltantes |
|---|---|---|
| `AvailabilityCheckerErrorsAlarm` | ≥1 error Lambda en 1 h; incluye timeout de función | `notBreaching` |
| `AvailabilityCheckerThrottlesAlarm` | ≥1 throttling Lambda en 1 h | `notBreaching` |
| `PipelineAdmissionErrorsAlarm` | ≥1 error de `pipeline-admission` en 1 h | `notBreaching` |
| `PipelineExecutionsFailedAlarm` | ≥1 fallo, timeout o aborto de Step Functions en 1 h | `notBreaching` |
| `OrchestrationRulesFailedInvocationsAlarm` | ≥1 fallo de invocación EventBridge en 1 h | `notBreaching` |
| `OpenF1UnavailableSustainedAlarm` | Solo creada para sondeos cada 15/30 min; 6 de 6 periodos horarios con `OpenF1Available < 1` | `notBreaching` |

El umbral de una incidencia para errores/throttling/fallos captura eventos
operativos concretos sin generar alarmas por ausencia de ejecuciones. La métrica
OpenF1 se emite en todas las cadencias, pero la alarma **no se crea** para
`rate(1 hour)`, `rate(3 hours)`, `rate(6 hours)`, `rate(12 hours)` ni
`rate(1 day)`: con esos intervalos no hay suficientes muestras horarias para
garantizar seis evaluaciones consecutivas. Para las cadencias de 15/30 minutos,
la alarma requiere seis horas continuas de indisponibilidad y evita alertar por
restricciones de sesión en vivo o rate limits breves. La ausencia de
`OpenF1Available` no significa indisponibilidad y no alarma. La regla de
EventBridge sigue `DISABLED` por defecto; no habrá muestras ni alertas mientras
esté deshabilitada, aunque se cree la alarma para una cadencia compatible.

`<ProjectName>-<Environment>-operations` contiene gráficos de resultados y
duración de Step Functions, errores/throttling y duración Lambda, resultados y
latencia OpenF1, fallos EventBridge y actividad/error/throttling de
`pipeline-admission`. Usa solo métricas y dimensiones presentes en este stack.
No hay un panel ni alarma de serving hasta que exista una integración de
despliegue que emita métricas HTTP compatibles.

El correo sigue siendo opcional. Crear `AlertEmailSubscription` con una
dirección solo significa que se solicitó la suscripción: estará
`PendingConfirmation` hasta que el destinatario acepte el correo de SNS. Solo
un `SubscriptionArn` confirmado representa notificaciones efectivas. Con
`AlertEmail` vacío, el topic y las alarmas existen, pero no hay destinatario.
La invocación asíncrona de `pipeline-admission` envía a SNS solo después de
agotar sus reintentos; la alarma `PipelineAdmissionErrorsAlarm` también publica
una notificación agregada por errores en la hora. Un mismo fallo final puede
generar ambos avisos: uno por invocación y otro por alarma, intencionalmente.
La métrica `Errors` cuenta **cada intento fallido**, también los que un reintento
resuelve después, así que un fallo transitorio puede disparar la alarma sin
llegar a `OnFailure`; un fallo definitivo puede producir hasta tres avisos
(`OnFailure`, ALARM y OK).
Step Functions mantiene su propia notificación de fallo, independiente de
estas dos rutas.

### Reserva, reintentos y ejecución manual

La escritura inicial del objeto usa `If-None-Match: *`; al vencer el intervalo
se reemplaza con `If-Match` del ETag leído. El mismo `check_id` y el mismo
payload reutilizan el `reserved_at` original y el nombre determinista de la
ejecución. Si `StartExecution` falla antes de ser aceptado, el mismo evento
puede volver a intentarlo sin mover la ventana. Si la respuesta es ambigua, el
reintento usa el mismo nombre y `ExecutionAlreadyExists` se trata como inicio
aceptado. Un nuevo `check_id` queda bloqueado durante `PipelineIntervalHours`,
también si el pipeline posteriormente falla; no hay liberación automática.

La reserva aplica solo a inicios automáticos desde la Lambda. Una persona o
automatización autorizada puede iniciar manualmente la state machine mediante
la consola de Step Functions o `aws stepfunctions start-execution`; esa
autorización manual no está protegida por el CAS y no se probó contra AWS en
esta validación local. La state machine sí conserva su chequeo de ejecución
concurrente activa.

## Planificación incremental e idempotencia M6 (R1)

El plan usa los catálogos de meetings/sessions guardados cuando todavía son
vigentes; los catálogos del año actual se refrescan después de 24 horas y
`--refresh-catalogs` permite pedir un refresco explícito de metadatos. No se
descargan datos de sesión al planificar. Cada sesión se incluye salvo que el
estado persistido `manifests/incremental/config=<fingerprint>/year=<año>/session=<id>/state.json` (layout lake; en el layout local, bajo `<manifest_path>/incremental/`, dentro de la raíz del almacén y nunca en el directorio de trabajo)
demuestre que M2–M5 finalizaron y que la configuración, los envelopes raw, los
reportes de etapa y los Parquet consolidados/features siguen coincidiendo con
sus hashes. Sidecars vinculan M3 con raw/M2, M4 con M3 y su salida, y M5 con
M4 y su salida; `--stage m5` aislado después de un cambio raw no puede
refrescar la completitud. Los HTTP 404 se aceptan como ausencia solo con una
entrada M2 fallida, no reintentable y correspondiente a la sesión; el estado
registra explícitamente esa ausencia. Perder o alterar la evidencia vuelve a
incluir la sesión. El estado respeta la raíz del `DataStore` local o S3 y se
separa por fingerprint de configuración. El plan guarda `sessions` para el Map de Step Functions y
`decisions` con motivo de inclusión u omisión. Un artefacto ausente, ilegible,
alterado o con procedencia no verificable fuerza reprocesamiento.

Medición sintética local: planificar 30 sesiones completadas con 500 filas
raw por sesión (1.312 MiB de raw en total) tardó 1.960 s y omitió las 30
sesiones. La verificación recalcula hashes
de envelopes raw, informes y Parquet en varios pasos (los Parquet y informes se
leen más de una vez). Es una medición local ilustrativa con datos sintéticos
diminutos: no demuestra el rendimiento sobre el histórico completo ni sobre
S3/Fargate, y no permite fijar el timeout de 1800 s. El coste remoto debe medirse
antes de depender del plan en producción.

Esta verificación detecta cambios en los bytes de los datos ya almacenados,
pero no puede detectar una corrección remota de OpenF1 sin volver a obtener el
contenido. Para una sesión histórica, el refresco del catálogo solo detecta
cambios de metadatos; para reevaluar contenido remoto se requiere una operación
explícita de M2 con `--overwrite`. Los hashes SHA-256 no se sustituyen por ETags.

M6 se reutiliza solo si el reporte más reciente es `VALID`, tiene el mismo
`input_fingerprint` y huellas de configuración, código y dependencias, y
`verify_report` confirma los ocho artefactos declarados. En cualquier duda se
ejecuta M6 nuevamente. La reutilización conserva el `run_id` y fingerprint del
reporte original; no crea un reporte falso para la nueva ejecución.

La función `pipeline-admission` utiliza una escritura condicional sobre un único
objeto S3 como compare-and-set: eventos concurrentes no pueden reservar el
mismo intervalo, y el estado sobrevive a reinicios de Lambda. Una reserva se
mantiene aunque Step Functions falle; por diseño, el siguiente inicio
automático se permite al cumplirse el intervalo, no en cada reintento de
disponibilidad. Una repetición del mismo evento/check ID conserva el timestamp
original, puede reintentar un inicio que falló antes de ser aceptado y usa un
nombre de ejecución determinista.
Los inicios manuales autorizados a `states:StartExecution` evitan esa compuerta
diaria y siguen sujetos a la comprobación de concurrencia de la state machine;
el calendario permanece `DISABLED` por defecto.

## Backfill histórico por fases (R2)

La primera carga (2023–2025) no cabe en una sola ejecución (límite de 12 h de la
state machine), así que se hace **una temporada por ejecución**.

**Cómo se acota una fase.** El input de la ejecución es el resultado de
disponibilidad; un inicio manual añade `extraction_years` (cadena, p. ej.
`"2024"` o `"2023,2024"`):

```
{"available": true, "extraction_years": "2024"}
```

`ResolveExtractionScope` copia ese valor a `$.scope.value` (si falta, usa el
parámetro de pila `ExtractionYears`, vacío por defecto = las temporadas de
`configs/config.yaml`; cualquier valor que no sea una cadena termina en
`InvalidExtractionScope` antes de lanzar ninguna tarea). `PlanSessions` lo recibe
como `F1_EXTRACTION_YEARS` en el contenedor, y el plan solo lista esas
temporadas. Las tareas M2–M5 reciben la sesión por línea de comando y no
dependen de esa variable. `years` no forma parte de la huella de configuración
del estado incremental, de modo que lo registrado por una fase lo ven las demás.

**Reanudación e idempotencia.** Se apoya en R1: tras una fase fallida, repetir la
misma ejecución solo planifica las sesiones sin evidencia verificada (las
completas se omiten sin escribir nada); repetir una temporada completa no
planifica ni escribe nada.

**Compuerta de M6.** La tarea M6 se lanza siempre con `--require-complete-history`.
Antes de modelar, `history_completeness` recorre cada temporada de
`modeling.years` y comprueba, solo con evidencia almacenada, que su catálogo
guardado lista al menos una sesión configurada y que **cada** sesión supera la
verificación de R1 (raw con hash, manifiesto M2, informes M3–M5 y salidas, y
configuración vigente). No hay ningún parámetro que la salte. Si falta algo, M6
termina con código **3** sin escribir artefactos de modelado; la state machine
va a `ModelingDeferredHistoryIncomplete` (ejecución *Succeeded*, sin alerta, y
sin pasar por `PipelineSucceeded`) y el motivo queda en los logs de la tarea
(`m6_deferred`, `backfill_history_status`, con las sesiones pendientes de cada
temporada). Cuando las tres temporadas están completas, M6 se ejecuta sobre
todo el dataset consolidado con sus guardas anti-leakage y su idempotencia
habituales. Esto cambia el semáforo de una ejecución completa: un `Succeeded`
en una fase intermedia **no** significa que el histórico esté completo.

**Secuencia de una carga completa**

1. Arrancar manualmente una ejecución con `{"available": true, "extraction_years": "2023"}`.
2. Comprobar que termina en `ModelingDeferredHistoryIncomplete`; repetir con `"2024"`.
3. Con `"2025"` el histórico queda completo y la misma ejecución termina en `PipelineSucceeded` tras ejecutar M6.
4. Si una fase falla, lanzar de nuevo la misma temporada.

**Límites de esta validación.** Las pruebas locales usan un cliente OpenF1
simulado, tres temporadas sintéticas de dos sesiones y la CLI real; no miden
duración, memoria ni coste. La duración de una temporada real (unas 22
carreras × 4 tareas en serie, con arranque de Fargate por tarea) no está
medida y debe confirmarse en la primera fase real antes de asumir que cabe en
las 12 h. Que ECS acepte `F1_EXTRACTION_YEARS` con valor vacío en el override
(el valor por defecto) no está verificado en AWS; el código lo trata como
«sin override».

## Tareas Fargate (Fase 1B)

Todas las etapas usan **la misma imagen y la misma task definition**; la state
machine solo cambia el comando (`python -m f1_race_intelligence …`). Cada
etapa lee sus entradas de S3 y escribe sus salidas en S3 (layout `lake`), así
que ninguna tarea depende del disco de otra.

| Estado | Comando | Lee | Escribe |
|---|---|---|---|
| `PlanSessions` | `--stage plan --plan-output s3://<bucket>/manifests/plan/<ejecución>.json` | Catálogos guardados; OpenF1 solo si faltan o requieren refresco (`meetings`, `sessions`) | `raw/…/endpoint=meetings|sessions/`, `manifests/extraction/`, decisiones del plan |
| `M2_Extraction` | `--stage m2 --year Y --meeting-key M --session-key S` | OpenF1 | `raw/year=Y/meeting=M/session=S/…`, `manifests/extraction/year=Y/session=S/` |
| `M3_Validation` | `--stage m3 --year Y --meeting-key M --session-key S` | `raw/`, `manifests/extraction/` | `validated/year=Y/session=S/` |
| `M4_Consolidation` | `--stage m4 --year Y --meeting-key M --session-key S` | `raw/`, `validated/` | `consolidated/year=Y/session=S/`, `manifests/consolidation/` |
| `M5_Features` | `--stage m5 --year Y --meeting-key M --session-key S` | `consolidated/` | `features/year=Y/session=S/`, `manifests/features/` |
| `M6_ModelingDataset` | `--stage m6` | `features/` | `modeling/split=…/runs/<run_id>/`, `modeling/preprocessing/runs/<run_id>/`, `manifests/modeling/` |

Las pruebas de `tests/infra/test_state_machine.py` recorren la definición,
generan esos comandos y los pasan por el parser real de la CLI.

**Ámbito completo de sesión (`--meeting-key`).** M2, M3, M4 y M5 reciben el mismo
`year`, `meeting_key` y `session_key`, tomados de la unidad de trabajo del plan
(`$.session.*`, la que el Map reparte desde `manifests/plan/`). M3, M4 y M5 no
lo recibían: sin el `meeting_key`, la CLI no puede registrar la procedencia de
etapa ni la finalización de la sesión (R1, «session scope is incomplete», solo
un aviso en el log), el reporte M3 queda archivado con `meeting_key` nulo y la
búsqueda por ámbito completo no lo encuentra. Consecuencia: ninguna sesión
procesada por la state machine podía quedar completa, el plan la reincluía en
cada ejecución y `--require-complete-history` de M6 la veía siempre incompleta.

La corrección solo **transmite el ámbito completo**. **No convierte una
validación FAIL en una sesión completada**: la procedencia de M3 se registra
únicamente con veredicto `PASS` o `WARNING`, y M4 y M5 exigen la de la etapa
anterior. Con M3 en `FAIL` (por ejemplo, los `n_gear` fuera de rango de las
sesiones reales 7953 y 9472) el comportamiento no cambia: M3 termina con código
0 (sin `--strict`), M4 y M5 pueden ejecutarse, no se escribe `state.json` y el
plan vuelve a incluir la sesión. Las reglas de M3 y el tratamiento de `n_gear`
no se tocan.

*Estado:* definido en `infra/statemachine/pipeline.asl.json`, cubierto por
tests locales (`tests/infra/test_state_machine.py` y
`tests/unit/test_incremental_e2e.py`, que ejecuta los comandos derivados del
ASL con la CLI real) y **desplegado en `dev` el 2026-11-21**: la definición
activa de la state machine ya pasa `--meeting-key` en M2–M5 (ver «Despliegue de
la corrección de `meeting_key`»). **No se ha ejecutado la state machine** con esa
definición, así que el registro de procedencia a través de ella sigue validado
solo en local.

**Resultado de ECS y códigos de salida (R3).** Ningún estado desreferencia
`Containers[0].ExitCode` al terminar la tarea: el resultado completo de
`runTask.sync` se guarda en su `ResultPath` y un `Choice` lo valida con
`IsPresent` + `IsNumeric` antes de comparar. Solo un `ExitCode` numérico igual a
0 continúa (M6: `PipelineSucceeded`); otro número es `…ExitedNonZero`; un
resultado sin `Containers`, con la lista vacía, sin `ExitCode`, nulo o no
numérico es `…ResultMalformed` (`StageResultMalformed`). Ambos casos terminan en
`NotifyPipelineFailure`. (Antes, un `ResultSelector` con esa ruta habría lanzado
`States.Runtime`, que un `Catch` no captura, sin aviso.) El código 3 de M6 sigue
siendo el único diferimiento controlado (`ModelingDeferredHistoryIncomplete`, sin
alerta); en cualquier otra etapa, 3 es un fallo. Las simulaciones locales
recorren estos caminos; cómo informa realmente ECS de un contenedor que no llegó
a arrancar solo se verifica en AWS.

**Endurecimiento**

- Imagen: sin ejecutables setuid/setgid (se retiran y se verifica en el build).
- Tarea: usuario `10001:10001`, `readonlyRootFilesystem: true`,
  `capabilities.drop: ["ALL"]`, sin puertos, sin secretos en el entorno.
- Sin volúmenes: con la raíz de solo lectura se comprobó en local que plan,
  M3–M6 y el modo AWS no escriben en ningún directorio (tampoco en `/tmp`).
  `tmpfs` no existe en Fargate; si una etapa futura necesitara espacio
  temporal, se añadiría un volumen de tarea montado en `/tmp`.

Estas medidas **mitigan** la explotación local; **no corrigen** los paquetes
vulnerables de la imagen base Debian 12 (sin parche publicado en bookworm).

**Hallazgos del escaneo de ECR (imagen `88514b4`)**

*Procedencia.* Verificado el 2026-11-21: escaneo de ECR del digest
`sha256:ad1d546424cdcf263020cd7813889f0c86b1aceccc2d501d49fbaa3803132799`
(base `python:3.12-slim-bookworm` con el digest anterior): **17 HIGH, 28 MEDIUM y
8 LOW**. El estado de parche procede del Debian Security Tracker (consultado el
mismo día); la presencia de cada biblioteca, de los ejecutables y de las
bibliotecas cargadas se comprobó dentro de la imagen. Sustituye a los 13
hallazgos (7 HIGH) reportados en el ROADMAP (D2), que no eran verificables
desde el repositorio.

*Remediación.* Subir el digest de la base a
`sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df`
(Python 3.12.16, misma distribución Bookworm) actualiza `apt` a 2.6.2 y
`openssl`/`libssl3` a `3.0.22-1~deb12u2`, y es el único cambio. No se usa
`apt-get upgrade` (no sería reproducible) ni se migra a Trixie.

*Resultado medido.* La imagen resultante (commit
`3d3b4e685f3456f33c1bf9ed169f826dcef93083`, etiqueta `3d3b4e6`, digest
`sha256:b4c0f5012acd02062d0bb236e99b4cd55ecc4545fe4054a35618ca0a80cd5fdf`) se
publicó en ECR y su escaneo terminó `COMPLETE` el 2026-11-21. Comparado con el de
`88514b4` (verificado en ECR, no estimado):

| Severidad | `88514b4` | `3d3b4e6` |
|---|---|---|
| HIGH | 17 | **15** |
| MEDIUM | 28 | **25** |
| LOW | 8 | **6** |
| Total | 53 | **46** |

Desaparecieron **exactamente siete hallazgos** y **no apareció ninguno nuevo**:
HIGH `apt` CVE-2026-109518 y `openssl` CVE-2026-84782; MEDIUM `openssl`
CVE-2026-35189, CVE-2026-75805 y CVE-2026-75806; LOW `openssl` CVE-2026-54872 y
CVE-2026-77696. Coincide con lo previsto a partir del Debian Security Tracker.

*Resumen de los 17 HIGH de `88514b4`:* **2 con parche disponible** (ya aplicado
en `3d3b4e6`) y **15 sin parche disponible en Bookworm**, que son los 15 HIGH que
siguen abiertos en `3d3b4e6`: 3 mitigados por configuración verificada y 12 que
requieren aceptación de riesgo. **Ninguno de los doce está aceptado todavía**:
la aceptación formal debe registrarla el responsable humano.

| CVE | Paquete (versión) | Estado en Debian | Tratamiento |
|---|---|---|---|
| CVE-2026-109518 | apt 2.6.1 | corregido: 2.6.2 (`bookworm-security`) | **Parche disponible**: lo aplica el nuevo digest |
| CVE-2026-84782 | openssl 3.0.22-1~deb12u1 | corregido: `~deb12u2` (`bookworm-security`) | **Parche disponible**: lo aplica el nuevo digest |
| CVE-2026-53613 | util-linux 2.38.1-5+deb12u3 | sin parche en Bookworm | **Mitigado por configuración verificada** |
| CVE-2026-78410 | util-linux (igual) | sin parche en Bookworm | **Mitigado por configuración verificada** |
| CVE-2026-84243 | glibc 2.36-9+deb12u14 | sin parche en Bookworm ni en Trixie | **Mitigado por configuración verificada** |
| CVE-2026-53615 | util-linux (igual) | sin parche en Bookworm | **Requiere aceptación** |
| CVE-2025-6020 | pam 1.5.2-6+deb12u2 | corregido solo en Trixie | **Requiere aceptación** |
| CVE-2026-112314, -112320, -112321, -112322, -112323, -119079 | krb5 1.20.1-2+deb12u5 | sin corregir en ninguna versión de Debian | **Requieren aceptación** (6) |
| CVE-2026-119373, -119374, -119379 | sqlite3 3.40.1-2+deb12u2 | sin parche en Bookworm ni en Trixie | **Requieren aceptación** (3) |
| CVE-2026-85091 | zlib 1:1.2.13.dfsg-1 | corregido solo en Trixie | **Pendiente: evidencia incompleta**; requiere aceptación mientras tanto |

*Mitigaciones de configuración (verificadas en la imagen y en la ejecución de la
tarea).* CVE-2026-53613 y -78410 exigen un `mount` setuid con una entrada
`fstab` de usuario: la imagen no tiene ningún fichero setuid/setgid (el build lo
verifica), la tarea se ejecuta como UID 10001 con `capabilities.drop: ["ALL"]` y
no monta nada. CVE-2026-84243 exige que un tercero fije `LANGUAGE`/`LANG` en el
entorno del proceso y que este cruce un límite de confianza sin `AT_SECURE`: el
entorno lo define la task definition (solo variables `F1_*`) y no hay ejecutables
setuid ni servicios que acepten variables de entorno de clientes.

*Riesgos que requieren aceptación (no son mitigaciones de configuración).* Para
`krb5`, `sqlite3`, `pam` y CVE-2026-53615 hay **evidencia de alcance**, no
mitigación: tras importar el paquete y sus dependencias de runtime no se cargan
`libgssapi`/`libkrb5`, `libsqlite3`, `libpam`, `libmount` ni `libblkid`; el lock
de runtime no incluye consumidores de sqlite, Kerberos ni PAM; no hay servicios
(`sshd`, servidores krb5, cron). Siguen sin parche en Bookworm y la aceptación
debe quedar registrada por el responsable humano. **zlib** (CVE-2026-85091) se
mantiene como riesgo pendiente: el módulo `zlib` de Python no referencia ninguna
función `gz*` (solo `deflate`/`inflate`), pero la comprobación no cubre código
nativo que enlace `libz` por su cuenta, así que la evidencia no se considera
completa.

*Limitaciones.* La lista de bibliotecas cargadas sale de importar los módulos del
proyecto, no de una traza exhaustiva de todas las rutas de ejecución. Los
resultados de las etapas M3–M5 en la imagen se probaron con datos sintéticos; M2
no se ha probado dentro de la imagen (requiere OpenF1). El escaneo de ECR es el
escáner básico del servicio: informa de presencia y versión de paquetes, no del
alcance de cada CVE.

**Red y costes**

VPC dedicada con subredes públicas e Internet Gateway; cada tarea recibe una
IP pública (`AssignPublicIp: ENABLED`) para salir por HTTPS a OpenF1, ECR, S3
y CloudWatch Logs. Se descartó NAT Gateway y VPC endpoints de interfaz porque
facturan por hora aunque no se ejecute nada. La VPC, subredes, tabla de rutas,
gateway y cluster no tienen coste propio; sí lo tienen la **IPv4 pública** de
cada tarea mientras corre, el cómputo Fargate (0,5 vCPU / 1 GB, mínimo de un
minuto por tarea), los logs y la transferencia de datos. El smoke del
2026-11-21 (ver «Validación de runtime AWS») comprobó que las tareas obtienen la
imagen de ECR, escriben en CloudWatch Logs y leen y escriben en S3. La
conectividad con OpenF1 no se ha probado; se verifica en la primera ejecución
controlada.

**Publicar y activar (fase posterior, con aprobación)**

1. Publicar la imagen y anotar su **digest de manifiesto**.
2. Desplegar con `PipelineImageUri=<repositorio>@sha256:<digest>` (añadirlo a
   la lista completa de parámetros de "Alertas por correo"), manteniendo
   `PipelineEnabled=false`, y revisar el change set.
3. Primera ejecución controlada **manual** de una sesión, iniciando la state
   machine mediante `StartExecution` con un input de disponibilidad válido
   (`{"available": true, ...}`); esta vía no consume ni espera el intervalo
   automático. Comprobar en el
   historial: que un exit code ≠ 0 se detecta (la documentación no aclara si
   `runTask.sync` falla el estado o termina con éxito; la definición cubre
   ambos casos), el formato de `Containers[0].ExitCode` en la salida, la
   respuesta de `listExecutions` (`Executions`) y que el ámbito IAM de
   `DescribeTasks`/`StopTask` basta.
4. Solo después, `PipelineEnabled=true`; el calendario sigue siendo una
   decisión aparte.

## Validación de runtime AWS (smoke del 2026-11-21)

Smoke sintético de las tareas Fargate en `dev` (`us-east-1`), ejecutado el
2026-11-21 con autorización explícita, operación por operación. **Procedencia:**
resultados medidos durante esa sesión con la AWS CLI y los logs de las tareas;
no se pueden comprobar desde el repositorio (AGENTS.md §6). Los ARN completos
contienen el ID de cuenta y no se versionan: las tareas se identifican por su ID.

Alcance: una tarea por prueba, lanzadas fuera de la state machine, con datos
**sintéticos** y un prefijo S3 aislado. No hubo llamadas a OpenF1, ni M2, ni
ejecuciones de Step Functions.

### A. Runtime de Fargate

| Elemento | Resultado |
|---|---|
| Task definition | revisión 1 |
| Imagen | por digest, `sha256:b4c0f5012acd02062d0bb236e99b4cd55ecc4545fe4054a35618ca0a80cd5fdf` |
| Prueba | una tarea con el comando `--help` |
| Resultado | exit code 0; la tarea terminó correctamente |
| Logs | 64 eventos recibidos en CloudWatch Logs |
| Endurecimiento observado | UID 10001, sistema de ficheros raíz de solo lectura y capacidades eliminadas |

### B. S3 y procesamiento sintético

| Tarea (ID) | Etapa | Resultado |
|---|---|---|
| `e11a42719ed045efac53cc14a1b5ce0a` | `copy-raw` | 12 archivos, 10.208 bytes, cero fallos; exit code 0 |
| `c71137e1ebea484ba8baec2e9b72eacd` | M3, M4 y M5 en una sola tarea | exit code 0 en las tres etapas |

- **M3:** 81 reglas, 78 superadas, 3 advertencias, cero errores. Conservó su
  estado `PARTIAL` (`WARNING`).
- **M4:** 6 filas de entrada y 6 de salida, cero pérdidas, 2 pilotos; 32.910 bytes.
- **M5:** 6 filas, 4 con objetivo (target), 51 features; 35.608 bytes.

### C. Evidencia en S3

- Prefijo: `smoke-tests/fargate-synthetic/20261121-856868e1/` del bucket de datos.
- 29 objetos, 127.361 bytes: 12 de entrada, 12 copias `raw` y 5 resultados derivados.
- Ningún objeto fuera de ese prefijo.
- El tamaño del resultado de M5 (35.608 bytes) coincide con el de la línea base
  local. **No se compararon hashes ni el contenido binario de los Parquet**: la
  coincidencia de tamaño no prueba que el contenido sea idéntico.
- Los 29 objetos se conservan; no se borró nada.

### D. Advertencias y limitaciones

- `incremental_stage_provenance_not_recorded`: esperada, porque no había un
  reporte histórico de M2 verificable.
- `consolidating_without_validation_pass`: M4 recibió una M3 con estado `WARNING`.
- Advertencia de NumPy por desviación cero en los datos sintéticos; no afectó a
  los resultados.
- No se ejecutó M2 ni hubo llamadas a OpenF1.
- No se probó la state machine real.

### E. Retirada del permiso temporal

Para el smoke se añadió al rol de tarea `…-data-processing` una declaración
`TemporarySyntheticFargateSmokeTest`: `s3:GetObject` y `s3:PutObject` solo sobre
el prefijo temporal. Se desplegó y después se retiró mediante un change set de
CloudFormation. Esa declaración nunca estuvo en `master` (vivió en una rama
temporal, ya eliminada).

- El change set se ejecutó tras aceptar expresamente tres cambios dependientes
  que CloudFormation marcó como dinámicos (`PipelineStateMachineRole`,
  `PipelineStateMachine` y `PipelineTaskDefinition`, que referencian el ARN del
  rol). Como el rol tiene `RoleName` fijo, su ARN no cambió y solo se actualizó
  `DataProcessingTaskRole`.
- El simulador de políticas de identidad confirmó `implicitDeny` para
  `GetObject` y `PutObject` en el prefijo temporal tras la retirada, y que las
  declaraciones originales no cambiaron: `GetObject`/`PutObject` siguen
  permitidos en las raíces operativas y `DeleteObject` sigue denegado.
- La task definition (revisión 1) y el digest de imagen no cambiaron. La
  definición de la state machine es idéntica byte a byte.
- **Límites de esta verificación:** se usó el simulador de políticas de
  identidad; no se hizo una prueba real de acceso tras la retirada ni se volvió a
  consultar la política del bucket.

### F. Estado final de AWS

| Elemento | Estado |
|---|---|
| Stack `f1-race-intelligence-dev` | `UPDATE_COMPLETE` |
| Task definition | revisión 1, `ACTIVE` |
| Alarmas | 6 de 6 en `OK` |
| Tareas ECS activas | 0 |
| Ejecuciones de Step Functions nuevas | ninguna |
| `PipelineEnabled` | `false` |
| `AvailabilityScheduleState` | `DISABLED` |
| Regla de calendario de EventBridge | `DISABLED` |

### Probado en AWS y solo validado en local

| Probado en AWS (con datos sintéticos) | Solo validado en local, no probado en AWS |
|---|---|
| Arranque de la imagen y task definition revisión 1 en Fargate | M2 y cualquier llamada a OpenF1 |
| Usuario 10001, raíz de solo lectura y capacidades eliminadas | `plan` y M6 (no se ejecutaron en Fargate) |
| `copy-raw`, M3, M4 y M5 leyendo y escribiendo S3 | La state machine real: exit codes vía `runTask.sync`, `listExecutions` y el ámbito IAM de `DescribeTasks`/`StopTask` |
| Logs en CloudWatch | Escrituras reales en las raíces operativas del rol de datos (solo se simularon con el simulador de IAM) |
| Retirada del permiso temporal (simulador de IAM) | Activación del pipeline y del calendario |
| Definición activa de la state machine con `--meeting-key` en M2–M5 (leída de Step Functions) | Una ejecución real de la state machine con esa definición (no se ha iniciado ninguna) |

## Despliegue de la corrección de `meeting_key` (2026-11-21)

**Procedencia:** medido con la AWS CLI durante el despliegue (lecturas de
CloudFormation, Step Functions, ECS e IAM); no se puede comprobar desde el
repositorio (AGENTS.md §6). `aws_status.json` no se ha modificado. Los ARN
completos contienen el ID de cuenta y no se versionan.

| Elemento | Resultado |
|---|---|
| Change set | `update-state-machine-meeting-key-20261121` (`d731a3d2-f46c-498c-a625-36e044a76ecd`), ejecutado el 2026-11-21 a las 20:08 hora local (UTC−5), es decir, 2026-11-22 01:08 UTC |
| Contenido | un único cambio: `Modify` de `PipelineStateMachine`, sin reemplazo, solo `DefinitionS3Location` y sin recursos dependientes |
| Resultado | stack `f1-race-intelligence-dev` en `UPDATE_COMPLETE`; solo `PipelineStateMachine` registró eventos de actualización |
| Definición activa | M2, M3, M4 y M5 incluyen `--meeting-key` con `$.session.meeting_key`; sus comandos son idénticos a los del ASL de `master`. Frente a la definición anterior hay 6 diferencias (3 comandos y 3 comentarios); estados, transiciones y el comando de M6 no cambian |
| Sin cambios | Task definition (revisión 1, la única `ACTIVE`, digest `sha256:b4c0f5012acd02062d0bb236e99b4cd55ecc4545fe4054a35618ca0a80cd5fdf`), imagen (`PipelineImageUri`), los 5 roles IAM (políticas inline, relación de confianza y ARN, comparados con una línea base tomada antes del cambio) y los 13 parámetros (`UsePreviousValue`) |
| Siguen deshabilitados | `PipelineEnabled=false`, `AvailabilityScheduleState=DISABLED` y la regla de calendario de EventBridge `DISABLED` |
| Actividad | 0 tareas ECS; ninguna ejecución nueva de Step Functions (la última es la del 2026-10-23); 6 de 6 alarmas en `OK` |

**Cómo se aplicó.** El change set se preparó a partir de la plantilla ya
desplegada, cambiando solo la clave `DefinitionS3Location` (el ASL corregido se
subió como artefacto de despliegue de SAM), y no desde la plantilla completa de
`master`, porque esta incluye M7 (ver «Pendientes abiertos», punto 2). No se usó
`sam deploy`.

**Alcance.** La definición está actualizada; no se ejecutó la state machine, no se
iniciaron tareas ECS ni se llamó a OpenF1. Esta corrección solo transmite el
ámbito completo de sesión: no convierte una validación `FAIL` en una sesión
completada (ver punto 3 de «Pendientes abiertos»).

## Validar (sin tocar AWS)

Requisitos: AWS SAM CLI y cfn-lint (`pip install aws-sam-cli cfn-lint`).

```bash
cd infra
sam validate --lint                       # SAM + cfn-lint
cfn-lint template.yaml
sam build                                 # empaqueta la Lambda en .aws-sam/
cd ..
pytest tests/infra                        # Lambda (servidor HTTP local), state machine, template
```

Validación por el servicio de CloudFormation (llama a la API de AWS, no crea
recursos; requiere credenciales):

```bash
aws cloudformation validate-template --template-body file://infra/template.yaml --region us-east-1
```

## Desplegar (más adelante)

```bash
# 0. Identidad y región correctas
aws sts get-caller-identity
aws configure get region

# 1. Construir y desplegar dev (muestra el change set y pide confirmación).
#    Los parámetros salen de samconfig.toml: calendario DISABLED, pipeline false.
cd infra
sam build
sam deploy --config-env dev

# 2. Ver outputs
aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev \
  --query "Stacks[0].Outputs" --output table

# 3. Probar la Lambda y ver la última ejecución de la state machine
aws lambda invoke --function-name f1-race-intelligence-dev-openf1-availability out.json && cat out.json
aws stepfunctions list-executions --max-results 5 \
  --state-machine-arn "$(aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev \
     --query "Stacks[0].Outputs[?OutputKey=='PipelineStateMachineArn'].OutputValue" --output text)"
```

Una invocación manual con `aws lambda invoke` es síncrona: **no** dispara los
destinos, así que no arranca la state machine. Para probar el flujo completo
sin esperar al schedule, invocar de forma asíncrona:

```bash
aws lambda invoke --function-name f1-race-intelligence-dev-openf1-availability \
  --invocation-type Event out.json
```

Para publicar la imagen batch en ECR (cuando corresponda):

```bash
# Solo desde un árbol limpio: cualquier cambio sin commit, incluidos archivos
# sin seguimiento, impide construir y publicar. La imagen registra el SHA
# completo del commit (F1_GIT_COMMIT) y F1_GIT_DIRTY=false.
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "Hay cambios sin commit: no se construye ni se publica la imagen." >&2
  false
else
  REPO=$(aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev \
    --query "Stacks[0].Outputs[?OutputKey=='PipelineImageRepositoryUri'].OutputValue" --output text)
  TAG=$(git rev-parse --short HEAD)
  aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}" &&
  docker build --platform linux/amd64 \
    --build-arg F1_GIT_COMMIT="$(git rev-parse HEAD)" \
    --build-arg F1_GIT_DIRTY=false \
    -t "$REPO:$TAG" . &&
  docker push "$REPO:$TAG" &&
  # Imagen de la Lambda de pronóstico (M7): mismo código y commit.
  FORECAST_REPO=$(aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev \
    --query "Stacks[0].Outputs[?OutputKey=='ForecastImageRepositoryUri'].OutputValue" --output text) &&
  docker build --platform linux/amd64 \
    --build-arg F1_GIT_COMMIT="$(git rev-parse HEAD)" \
    --build-arg F1_GIT_DIRTY=false \
    -f Dockerfile.forecast -t "$FORECAST_REPO:$TAG" . &&
  docker push "$FORECAST_REPO:$TAG"
fi
```

`--platform linux/amd64` coincide con la task definition (`X86_64`). La
etiqueta corta solo nombra la imagen; el commit exacto queda en la imagen y
en los reportes de M6. El digest que se pasa en `PipelineImageUri` es el que
devuelve `docker push`.

## Alertas por correo (`AlertEmail`)

El topic `…-critical-alerts` recibe las alarmas de CloudWatch y los fallos de la
Lambda (`OnFailure`). Que alguien se entere depende de `AlertEmail`:

- **Vacío** (valor por defecto y el de `samconfig.toml`): la condición
  `HasAlertEmail` es falsa y no se crea ninguna `AWS::SNS::Subscription`. El
  topic existe y recibe mensajes, pero **nadie los recibe**: las alertas no son
  operativas.
- **Con una dirección real**: CloudFormation crea una suscripción `email` y AWS
  envía al destinatario un correo *AWS Notification - Subscription
  Confirmation*. Mientras no se pulse el enlace, la suscripción queda en
  `PendingConfirmation` y **no se entrega ninguna alerta**. Las notificaciones
  solo cuentan como operativas cuando el destinatario la confirma.

La dirección no se guarda en `samconfig.toml` ni en ningún archivo del
repositorio: se pasa al desplegar. `--parameter-overrides` en la línea de
comandos **sustituye por completo** la lista de `samconfig.toml` (no la
combina), y un parámetro omitido toma el `Default` de la plantilla, no el valor
desplegado. Por eso el comando repite todos los parámetros del entorno,
calendario `DISABLED` incluido.

Windows PowerShell (5.1 o 7), desde la raíz del repositorio. Todo va en un
único bloque para que, si falta la dirección, no se ejecute ningún comando de
SAM ni aunque se pegue en la consola:

```powershell
# $env:ALERT_EMAIL debe contener la dirección real del destinatario (no un ejemplo):
#   $env:ALERT_EMAIL = Read-Host 'Correo para las alertas'
if ([string]::IsNullOrWhiteSpace($env:ALERT_EMAIL) -or $env:ALERT_EMAIL -notmatch '^[^@\s]+@[^@\s]+\.[^@\s]+$') {
    Write-Error 'Define $env:ALERT_EMAIL con la dirección real del destinatario. No se ha ejecutado SAM.'
} else {
    Push-Location infra
    try {
        sam build
        if ($LASTEXITCODE -eq 0) {
            sam deploy --config-env dev --parameter-overrides `
                "ParameterKey=ProjectName,ParameterValue=f1-race-intelligence" `
                "ParameterKey=Environment,ParameterValue=dev" `
                "ParameterKey=AvailabilitySchedule,ParameterValue='rate(15 minutes)'" `
                "ParameterKey=AvailabilityScheduleState,ParameterValue=DISABLED" `
                "ParameterKey=PipelineIntervalHours,ParameterValue=24" `
                "ParameterKey=PipelineEnabled,ParameterValue=false" `
                "ParameterKey=LogRetentionDays,ParameterValue=30" `
                "ParameterKey=AlertEmail,ParameterValue=$env:ALERT_EMAIL"
        }
    } finally {
        Pop-Location
    }
}
```

En PowerShell, `rate(15 minutes)` va entre comillas **simples** dentro del
argumento: SAM las acepta, y Windows PowerShell 5.1 no transmite bien comillas
dobles anidadas a un ejecutable nativo (`sam` es un `.cmd`). Para solo preparar
el change set sin aplicarlo, añadir `--no-execute-changeset` a `sam deploy`.

Bash:

```bash
# ALERT_EMAIL debe contener la dirección real del destinatario (no un ejemplo).
: "${ALERT_EMAIL:?Define ALERT_EMAIL con la dirección real del destinatario}"
cd infra
sam build
sam deploy --config-env dev --parameter-overrides \
  ParameterKey=ProjectName,ParameterValue=f1-race-intelligence \
  ParameterKey=Environment,ParameterValue=dev \
  'ParameterKey=AvailabilitySchedule,ParameterValue="rate(15 minutes)"' \
  ParameterKey=AvailabilityScheduleState,ParameterValue=DISABLED \
  ParameterKey=PipelineIntervalHours,ParameterValue=24 \
  ParameterKey=PipelineEnabled,ParameterValue=false \
  ParameterKey=LogRetentionDays,ParameterValue=30 \
  "ParameterKey=AlertEmail,ParameterValue=${ALERT_EMAIL}"
```

En ambos casos `rate(15 minutes)` va entrecomillado dentro del argumento porque
SAM separa los valores por espacios: sin comillas se queda con `rate(15`, que no
está en `AllowedValues`. Para `prod`: `--config-env prod`, `Environment=prod` y
`LogRetentionDays=90`.

Antes de aceptar el change set, comprobar que no aparece ningún cambio en
`AvailabilityScheduleRule` y que el único recurso nuevo es
`AlertEmailSubscription`. Después del despliegue y de que el destinatario
confirme:

```bash
aws sns list-subscriptions-by-topic --topic-arn "$(aws cloudformation describe-stacks \
  --stack-name f1-race-intelligence-dev \
  --query "Stacks[0].Outputs[?OutputKey=='AlertTopicArn'].OutputValue" --output text)" \
  --query "Subscriptions[].SubscriptionArn" --output text
# Un ARN completo = confirmada; "PendingConfirmation" = todavía no llegan alertas.
```

```powershell
$topic = aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev `
    --query "Stacks[0].Outputs[?OutputKey=='AlertTopicArn'].OutputValue" --output text
aws sns list-subscriptions-by-topic --topic-arn $topic --query "Subscriptions[].SubscriptionArn" --output text
# Un ARN completo = confirmada; "PendingConfirmation" = todavía no llegan alertas.
```

Activar el calendario (`AvailabilityScheduleState=ENABLED`) no forma parte de
ningún ejemplo de este documento: es una decisión explícita y revisada, que se
toma sustituyendo `DISABLED` en el despliegue correspondiente.

## Modelo de carrera (M7): desplegar y mantener

El modelo (`src/f1_race_intelligence/race_model/`, matemática en
`docs/f1-model-math.md`) se despliega en dos piezas, cada una creada **solo cuando su
imagen ya está en ECR** (parámetros vacíos ⇒ el stack no crea nada de M7 salvo el
repositorio ECR del pronóstico, los roles y los log groups):

| Pieza | Recurso | Se crea si | Qué hace |
|---|---|---|---|
| Entrenamiento / actualización | ECS Fargate (`RaceModelTaskDefinition`, imagen batch, `PipelineCluster`) + Step Functions `…-race-model` | `PipelineImageUri` no vacío | `{"action":"train"}`: `m7-train` → `m7-evaluate --promote`. `{"action":"update"}`: `m7-update` |
| Pronóstico | Lambda imagen `…-race-forecast` (`Dockerfile.forecast`) | `ForecastImageUri` no vacío | `{"session_key":…, "lap":…}` o `{"mode":"live"}` → `forecasts/m7/…` |

Datos en el lake (contrato, ver `storage/layout.py`):

```
models/m7/version=<v>/model.json        versión inmutable (JSON: parámetros + estado del filtro)
models/m7/latest.json                   puntero a la versión promovida (lo único que leen los pronósticos)
manifests/models/m7/*.json              reportes de train / evaluate / update
forecasts/m7/year=<y>/session=<s>/lap=<k>.json   y latest.json por carrera
```

IAM: `…-race-model` lee `raw/` y escribe solo `models/m7/` y
`manifests/models/m7/`; `…-race-forecast` lee `raw/` y `models/m7/` y escribe
solo `forecasts/m7/`. Ninguno puede borrar.

### Ciclo de vida

1. **Datos.** M2 ahora también extrae `session_result` (etiqueta) y
   `starting_grid` (prior de clasificación, pedido por `meeting_key`).
2. **Entrenar** (mensual o a mano): ajusta nivel 1 en todas las carreras, el filtro
   de temporada, eventos y baselines en las de entrenamiento, y escribe una versión
   nueva. `m7-evaluate --promote` la puntúa en validación y **solo** mueve
   `latest.json` si mejora el RPS medio del baseline `persist`; si no, la tarea sale
   con código 1, la ejecución falla y llega una alerta, y `latest` no cambia.
3. **Actualizar** (lunes tras cada GP): `m7-update` vuelve a correr el filtro con
   los hiperparámetros congelados para que la próxima carrera use el último
   resultado; promueve automáticamente (la versión padre ya estaba aprobada).
4. **Pronosticar**: invocación bajo demanda o la regla `…-race-forecast-live`
   (cada minuto; fuera de carrera responde `idle`).
5. **Revertir**: escribir la versión anterior en `models/m7/latest.json`
   (`aws s3 cp` de un `latest.json` con `{"version": "<v>"}`); las versiones nunca
   se sobrescriben.

### Publicar imágenes y desplegar

Las dos imágenes (batch y pronóstico) se construyen y publican con el bloque
protegido de "Para publicar la imagen batch en ECR": solo desde un árbol limpio
y con el SHA completo del commit. Las tareas del modelo usan **la misma imagen
batch** (`PipelineImageUri`), el mismo cluster, las mismas subredes y el mismo
security group que M2-M6; la Lambda usa `ForecastImageUri`. Ambas van **por
digest** (el que devuelve `docker push`), nunca por tag.

```bash
: "${ALERT_EMAIL:?Define ALERT_EMAIL con la dirección real del destinatario}"
: "${PIPELINE_IMAGE:?Define PIPELINE_IMAGE=<repositorio>@sha256:<digest> de la imagen batch}"
: "${FORECAST_IMAGE:?Define FORECAST_IMAGE=<repositorio>@sha256:<digest> de la imagen de pronóstico}"
cd infra && sam build
sam deploy --config-env dev --parameter-overrides \
  ParameterKey=ProjectName,ParameterValue=f1-race-intelligence \
  ParameterKey=Environment,ParameterValue=dev \
  'ParameterKey=AvailabilitySchedule,ParameterValue="rate(15 minutes)"' \
  ParameterKey=AvailabilityScheduleState,ParameterValue=DISABLED \
  ParameterKey=PipelineIntervalHours,ParameterValue=24 \
  ParameterKey=PipelineEnabled,ParameterValue=false \
  ParameterKey=LogRetentionDays,ParameterValue=30 \
  "ParameterKey=AlertEmail,ParameterValue=${ALERT_EMAIL}" \
  "ParameterKey=PipelineImageUri,ParameterValue=${PIPELINE_IMAGE}" \
  "ParameterKey=ForecastImageUri,ParameterValue=${FORECAST_IMAGE}" \
  ParameterKey=ModelScheduleState,ParameterValue=DISABLED \
  ParameterKey=LiveForecastScheduleState,ParameterValue=DISABLED
```

`PipelineEnabled=false` sigue apagando M2-M6: el modelo corre con su propia
state machine, que solo arranca a mano o con sus calendarios (`DISABLED`). En
PowerShell, el mismo despliegue sigue el patrón protegido de "Alertas por
correo" añadiendo `PipelineImageUri`, `ForecastImageUri`, `ModelScheduleState`
y `LiveForecastScheduleState`.

Primera carga del modelo y pruebas (los datos `raw/` deben estar en el bucket):

```bash
out() { aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
SM=$(out RaceModelStateMachineArn)
aws stepfunctions start-execution --state-machine-arn "$SM" --input '{"action": "train"}'
FN=$(out ForecastFunctionName)
aws lambda invoke --function-name "$FN" --cli-binary-format raw-in-base64-out \
  --payload '{"session_key": 9998, "lap": 30}' forecast.json && cat forecast.json
```

Local, sin AWS (mismos comandos que ejecutan las tareas):

```bash
python -m f1_race_intelligence --stage m7-train,m7-evaluate --promote
python -m f1_race_intelligence --stage m7-forecast --year 2025 --session-key 9998 --lap 30
python -m f1_race_intelligence --stage m7-evaluate --split test --unseal-test   # una sola vez, informe final
```

### Riesgos del modelo en producción

- **Datos en vivo.** Durante una sesión OpenF1 sirve datos en vivo solo a usuarios
  autenticados; sin esa credencial el modo `live` falla o ve datos atrasados. Por
  eso `LiveForecastScheduleState` es `DISABLED` y el cliente aún no envía token.
- **Puerta de promoción.** Promueve contra `persist`. En la validación de
  2025 el modelo supera claramente a `persist` y queda por delante de la
  matriz de transición (`transition`) sin diferencia significativa (IC 95%
  del ΔRPS que incluye 0); si se quiere exigir ambos, cambiar
  `race_model.promote_requires_beating`.
- **Test sellado.** Mientras `race_model.test_sealed` sea `true`, ni el entrenamiento, ni
  `m7-update`, ni los pronósticos leen carreras de test. `--split test --unseal-test` es
  la única entrada: no promueve, queda en `manifests/models/m7/test_ledger.json` y se
  rechaza la segunda vez. Con el test sellado, `m7-update` salta esas carreras; pasar
  `test_sealed: false` solo tras el informe final.
- El reinicio reglamentario de 2026 deja el filtro sin historia útil: tras
  `m7-update`, revisar `prior_check` en el reporte antes de confiar en las
  primeras carreras.

## Eliminar

`sam delete --stack-name f1-race-intelligence-dev` borra el stack pero
**conserva** el bucket y el repositorio ECR (`DeletionPolicy: RetainExceptOnCreate`); se
eliminan a mano si realmente se quiere perder los datos. La única excepción es
el rollback de la primera creación del stack: ahí se borran (aún vacíos) para
que el reintento no choque con sus nombres fijos.

## Riesgos conocidos

- El calendario está desactivado por defecto. Si se activa con el chequeo cada
  15 min y `PipelineEnabled=false`, cada chequeo invoca `pipeline-admission`,
  pero no inicia Step Functions. Con el pipeline habilitado, el intervalo S3
  permite como máximo un inicio automático por periodo configurado.
- Con `AlertEmail` vacío, o con la suscripción sin confirmar, las alarmas y los
  fallos de la Lambda no llegan a nadie.
- La Lambda sale a internet sin VPC. Si más adelante se mete en una VPC hará
  falta NAT.
- El topic SNS no usa KMS: las alarmas de CloudWatch no pueden publicar en un
  topic cifrado con la clave administrada `aws/sns`; cifrarlo requiere una CMK
  con política para CloudWatch.
- `session_key=7953` es el objetivo del sondeo: si OpenF1 lo retirara, el
  chequeo daría 404 → `openf1_error`. Cambiarlo es editar
  `OPENF1_PROBE_PATH`.
- El rol de datos da `ListBucket` sobre todo el bucket (ver arriba).

## Pendientes abiertos (2026-11-21)

Tres asuntos independientes entre sí. Ninguno se resuelve en este cambio.

### 1. Aviso W3005 en `ForecastFunctionLogGroup`

`sam validate --lint` sobre la plantilla de `master` (`800931b`) termina con
código 1 por un único aviso, `W3005`: `ForecastFunctionLogGroup` figura en
`DependsOn` de `ForecastFunction`, pero esa dependencia ya la impone el `Ref` de
`LoggingConfig.LogGroup`. Lo introdujo el PR #22 (M7); la plantilla anterior
pasaba el lint. No tiene efecto funcional, pero DEFINITION_OF_DONE exige que el
lint pase, así que la plantilla de `master` no puede darse por validada con él.
Pendiente: retirar ese `DependsOn` en un cambio aparte.

### 2. Divergencia entre la plantilla desplegada y `master` (M7)

- **Desplegado:** 44 recursos y 13 parámetros (la plantilla enviada al stack).
- **`master`:** el PR #22 añade 19 recursos, 3 salidas y 5 parámetros
  (`ModelTrainingSchedule`, `ModelUpdateSchedule`, `ModelScheduleState`,
  `LiveForecastScheduleState`, `ForecastImageUri`).
- **Con los parámetros actuales** (`PipelineImageUri` no vacío,
  `ForecastImageUri` vacío), un despliegue de `master` crearía 14 de esos
  recursos: 7 incondicionales (repositorio ECR del pronóstico, 3 log groups y
  3 roles) y 7 por `HasPipelineImage` (task definition, rol y state machine del
  modelo, rol de calendario, 2 reglas de calendario y una alarma). Los 5 de la
  Lambda de pronóstico requieren `ForecastImageUri`. Las reglas de calendario del
  modelo y del pronóstico se crean con estado `DISABLED` por defecto. Este
  recuento sale de leer la plantilla y sus condiciones, **no de un change set**.
- **Imagen:** la desplegada (`sha256:b4c0f501…5fdf`) es anterior al PR #22
  (registrada como construida desde `3d3b4e6`, dato reportado del registro de
  imágenes del ROADMAP externo), así que no incluye las etapas de M7 ni los
  endpoints nuevos de M2 descritos en «Modelo de carrera (M7)».
- **Mientras tanto:** los cambios a la state machine se aplican con change sets
  mínimos sobre la plantilla desplegada, como el del 2026-11-21; un `sam deploy`
  directo desde `master` no debe usarse sin revisar antes su change set.
  Alinear el stack con `master` (o no) es decisión del responsable humano.

### 3. Sesiones reales 7953 y 9472: M3 en `FAIL` por `n_gear`

| Sesión | M3 | Hallazgos | Evidencia |
|---|---|---|---|
| 2023 / meeting 1141 / **7953** | `FAIL` | 2 errores `values`: `n_gear` fuera de 0–8 en los pilotos 4 (32 registros) y 81 (31) | Fargate (2026-11-21): 113 reglas, informe `validated/year=2023/session=7953/validation_report_20261121T232645786678Z.json`; local con el código actual: 120 reglas |
| 2024 / meeting 1229 / **9472** | `FAIL` | 1 error `values`: `n_gear` en el piloto 2 (54 registros); 1 aviso `missing_values`: `pit.stop_duration` nulo al 100 % | Local con el código actual: 120 reglas (no está en S3) |

Los hallazgos de ambas sesiones coinciden con los del informe local del
2026-09-09. Los valores fuera de rango son ráfagas de 8 a 13 segundos antes de la
salida, con el coche parado (velocidad 0, rpm ≈ 4000) y marchas crecientes de 9
a 75, en el 0,09–0,16 % de las filas de cada piloto afectado. M4 enmascara esas
muestras (`n_gear <= MAX_GEAR`).

Consecuencia: con M3 en `FAIL` no se registra procedencia, no se escribe
`state.json` y la sesión no se completa, de modo que el plan la reincluye y
`--require-complete-history` de M6 la ve incompleta. Ni las reglas de M3 ni el
tratamiento de `n_gear` se han modificado. En AWS no se ha ejecutado M4 ni M5
sobre estas sesiones (solo M2 y M3 sobre la 7953); en local, M4 y M5 sí corrieron
el 2026-09-09 sobre las dos con M3 en `FAIL`, sin perder filas. Pendiente de
decisión: si el glitch debe seguir siendo un error de M3.

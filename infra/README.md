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
│   └── availability_checker/
│       ├── app.py                         # Lambda: sondeo liviano a OpenF1 (solo stdlib)
│       └── requirements.txt               # vacío a propósito
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
                     ▼                                 │   · errores de la Lambda
      Step Functions STANDARD  <p>-<env>-pipeline      │   · ejecuciones fallidas/timeout/abortadas
        LoadContext                                    │   · FailedInvocations de las reglas
        IsAvailabilityResultValid ── no ─► Fail        │   · OpenF1 no disponible 6 h seguidas (EMF)
        IsOpenF1Available ── false ─► OpenF1Unavailable (Succeed)
        IsPipelineEnabled ── false ─► PipelineDisabled (Succeed)   ← valor por defecto
        CheckForConcurrentRun ── otra ejecución RUNNING ─► AnotherRunInProgress (Succeed)
        PlanSessions        Fargate --stage plan --plan-output s3://…/manifests/plan/<ejecución>.json
        LoadPlan            s3:getObject del plan → $.plan.sessions
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

Por qué la Lambda no llama a Step Functions: se limita a responder. Su
resultado viaja por el *destination* OnSuccess a un bus de EventBridge, y una
regla arranca la state machine con ese JSON. Así la Lambda no tiene permisos
para iniciar nada y cada chequeo queda registrado como ejecución, que decide
explícitamente si termina (`available == false`) o sigue.

## Recursos

| Recurso | Configuración |
|---|---|
| S3 `DataBucket` | versionado; SSE-S3; bloqueo total de acceso público; `BucketOwnerEnforced`; política que niega tráfico sin TLS; lifecycle: aborta multipart a 7 d, expira versiones no actuales (parámetro, 90 d), limpia delete markers, `raw/` a Intelligent-Tiering a 30 d, `manifests/` a 90 d, `logs/` expira a 365 d; **nunca expira datasets actuales**; `DeletionPolicy: RetainExceptOnCreate`, `UpdateReplacePolicy: Retain` |
| Lambda `openf1-availability` | Python 3.12 arm64, 128 MB, 15 s; sin VPC; sin reintentos asíncronos; destinos OnSuccess→bus, OnFailure→SNS; métricas EMF `OpenF1Available`, `OpenF1LatencyMs` |
| EventBridge | regla programada (parámetros `AvailabilitySchedule`, `AvailabilityScheduleState`); bus propio + regla de resultados |
| Step Functions | STANDARD; logs `ALL` a CloudWatch; timeout global 12 h; plan y M2–M6 como `ecs:runTask.sync` (ver "Tareas Fargate") |
| ECS | cluster sin Container Insights; task definition Fargate `X86_64`, 512 CPU / 1024 MB, usuario `10001:10001`, raíz de solo lectura, `capabilities.drop: [ALL]`, sin puertos ni volúmenes; solo si `PipelineImageUri` tiene valor |
| Red | VPC dedicada `10.42.0.0/24`, 2 subredes públicas en 2 AZ, Internet Gateway, security group sin entrada y salida solo TCP 443 |
| SNS | topic de alertas críticas; política para CloudWatch de la misma cuenta; suscripción e-mail opcional |
| ECR | tags inmutables, escaneo al subir, expira sin tag a 7 d, conserva 30 imágenes; `RetainExceptOnCreate` |
| CloudWatch | 3 log groups con retención `LogRetentionDays`; 4 alarmas base y una condicional; dashboard operativo |

## IAM (mínimo privilegio)

| Rol | Confía en | Permisos |
|---|---|---|
| `…-availability-lambda` | `lambda.amazonaws.com` | escribir en su log group; `events:PutEvents` en el bus de orquestación; `sns:Publish` en el topic de alertas. **Sin S3.** |
| `…-pipeline-states` | `states.amazonaws.com` | `sns:Publish` en el topic; APIs de entrega de logs (`Resource: "*"` exigido por AWS); `ecs:RunTask` solo de la familia `<p>-<env>-pipeline` y con condición `ecs:cluster`; `ecs:DescribeTasks`/`StopTask` solo sobre tareas del cluster; `events:PutRule/PutTargets/DescribeRule` solo sobre la regla administrada `StepFunctionsGetEventsForECSTaskRule`; `iam:PassRole` solo de los dos roles de tarea y con `iam:PassedToService=ecs-tasks.amazonaws.com`; `s3:GetObject` solo en `manifests/plan/*`; `states:ListExecutions` solo sobre esta state machine |
| `…-availability-to-states` | `events.amazonaws.com` | `states:StartExecution` solo sobre la state machine |
| `…-data-processing` (tarea) | `ecs-tasks.amazonaws.com` | `s3:ListBucket` del bucket; `GetObject`/`PutObject` en `raw/ validated/ consolidated/ features/ modeling/ manifests/`. Sin `Delete*`, sin `models/`. |
| `…-data-processing-exec` (ejecución) | `ecs-tasks.amazonaws.com` | pull de la imagen del repositorio ECR (`GetAuthorizationToken` requiere `*`); escribir en `/aws/ecs/…` |

Todas las relaciones de confianza llevan `aws:SourceAccount`, salvo la del rol
de la Lambda, que usa la confianza estándar de `lambda.amazonaws.com`: no se
pudo verificar que Lambda admita esa condición en roles de ejecución, y si no la
admite `CreateFunction` podría fallar. No hay políticas administradas de AWS ni
`AdministratorAccess`.

`ecs:DescribeTasks`/`StopTask` se limitan a `task/<cluster>/*` aunque la
política que genera AWS usa `"*"`; la condición `ecs:cluster` y ese ámbito se
verifican en la primera ejecución controlada.

`s3:ListBucket` no se restringe por `s3:prefix` a propósito: sin ese permiso,
`HeadObject` sobre una clave inexistente devuelve 403 en vez de 404 y la
idempotencia del pipeline ("¿ya existe este archivo?") dejaría de funcionar.

## Parámetros

| Parámetro | Por defecto | Nota |
|---|---|---|
| `ProjectName` | `f1-race-intelligence` | prefijo de nombres y tag |
| `Environment` | `dev` | `dev` · `staging` · `prod` |
| `S3BucketName` | vacío | vacío ⇒ `<project>-<env>-<account>-<region>` |
| `AvailabilitySchedule` | `rate(15 minutes)` | solo valores ≥ 15 min; la alarma sostenida solo se crea para 15/30 min |
| `AvailabilityScheduleState` | `DISABLED` | `samconfig.toml` también lo fija en `DISABLED` (dev y prod); activarlo es una decisión explícita |
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

Los logs se escriben en los tres grupos descritos arriba y retienen el periodo
configurado. No se registran secretos, credenciales ni contenido de artefactos.
Los IDs de ejecución, `run_id`, `input_fingerprint`, estados y rutas de
artefactos son metadatos de diagnóstico, no sus contenidos. La ejecución M7 /
MLflow y el prototipo M8 no forman parte de esta infraestructura desplegable;
no hay métricas AWS de entrenamiento, publicación ni HTTP 5xx del serving.

| Métrica | Namespace / dimensión | Unidad y estadística | Significado |
|---|---|---|---|
| `ExecutionsStarted`, `ExecutionsSucceeded`, `ExecutionsFailed`, `ExecutionsTimedOut`, `ExecutionsAborted` | `AWS/States`, `StateMachineArn=<pipeline>` | Count / Sum | Volumen y desenlace de ejecuciones |
| `ExecutionTime` | `AWS/States`, `StateMachineArn=<pipeline>` | Milliseconds / Average, Maximum | Duración de las ejecuciones |
| `Invocations`, `Errors`, `Throttles` | `AWS/Lambda`, `FunctionName=<availability checker>` | Count / Sum | Actividad, errores de ejecución y limitación de concurrencia |
| `Duration` | `AWS/Lambda`, `FunctionName=<availability checker>` | Milliseconds / Average, Maximum | Duración de la Lambda |
| `FailedInvocations` | `AWS/Events`, `RuleName`; para la regla de resultados también `EventBusName` | Count / Sum | Fallos al invocar la Lambda o iniciar Step Functions |
| `OpenF1Available` | `f1-race-intelligence/<env>`, `Environment=<env>` | Count / Maximum (0 o 1) | Resultado del sondeo, emitido por EMF |
| `OpenF1LatencyMs` | `f1-race-intelligence/<env>`, `Environment=<env>` | Milliseconds / Average | Latencia del sondeo, emitida por EMF |

`<env>` y el prefijo del namespace se generan desde `ProjectName` y
`Environment`. Las métricas `AWS/*` son métricas de servicio; las dos
`OpenF1*` son las únicas métricas personalizadas. La Lambda emite EMF a su log
group y no necesita `cloudwatch:PutMetricData`.

### Alarmas y dashboard

Las cuatro alarmas base y, cuando la cadencia lo permite, la alarma sostenida
publican al mismo `AlertTopic`; las acciones `OKActions` también notifican la
recuperación:

| Alarma | Criterio | Datos faltantes |
|---|---|---|
| `AvailabilityCheckerErrorsAlarm` | ≥1 error Lambda en 1 h; incluye timeout de función | `notBreaching` |
| `AvailabilityCheckerThrottlesAlarm` | ≥1 throttling Lambda en 1 h | `notBreaching` |
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
latencia OpenF1 y fallos EventBridge. Usa solo métricas y dimensiones presentes
en este stack. No hay un panel ni alarma de serving hasta que exista una
integración de despliegue que emita métricas HTTP compatibles.

El correo sigue siendo opcional. Crear `AlertEmailSubscription` con una
dirección solo significa que se solicitó la suscripción: estará
`PendingConfirmation` hasta que el destinatario acepte el correo de SNS. Solo
un `SubscriptionArn` confirmado representa notificaciones efectivas. Con
`AlertEmail` vacío, el topic y las alarmas existen, pero no hay destinatario.
La Lambda también tiene destino `OnFailure` a SNS y Step Functions publica
detalles en `NotifyPipelineFailure`; por tanto, un mismo fallo puede generar
tanto la notificación directa como la de la alarma agregada.

## Tareas Fargate (Fase 1B)

Todas las etapas usan **la misma imagen y la misma task definition**; la state
machine solo cambia el comando (`python -m f1_race_intelligence …`). Cada
etapa lee sus entradas de S3 y escribe sus salidas en S3 (layout `lake`), así
que ninguna tarea depende del disco de otra.

| Estado | Comando | Lee | Escribe |
|---|---|---|---|
| `PlanSessions` | `--stage plan --plan-output s3://<bucket>/manifests/plan/<ejecución>.json` | OpenF1 (`meetings`, `sessions`) | `raw/…/endpoint=meetings|sessions/`, `manifests/extraction/`, el plan |
| `M2_Extraction` | `--stage m2 --year Y --meeting-key M --session-key S` | OpenF1 | `raw/year=Y/meeting=M/session=S/…`, `manifests/extraction/year=Y/session=S/` |
| `M3_Validation` | `--stage m3 --year Y --session-key S` | `raw/`, `manifests/extraction/` | `validated/year=Y/session=S/` |
| `M4_Consolidation` | `--stage m4 --year Y --session-key S` | `raw/`, `validated/` | `consolidated/year=Y/session=S/`, `manifests/consolidation/` |
| `M5_Features` | `--stage m5 --year Y --session-key S` | `consolidated/` | `features/year=Y/session=S/`, `manifests/features/` |
| `M6_ModelingDataset` | `--stage m6` | `features/` | `modeling/split=…/runs/<run_id>/`, `modeling/preprocessing/runs/<run_id>/`, `manifests/modeling/` |

Las pruebas de `tests/infra/test_state_machine.py` recorren la definición,
generan esos comandos y los pasan por el parser real de la CLI.

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

**Red y costes**

VPC dedicada con subredes públicas e Internet Gateway; cada tarea recibe una
IP pública (`AssignPublicIp: ENABLED`) para salir por HTTPS a OpenF1, ECR, S3
y CloudWatch Logs. Se descartó NAT Gateway y VPC endpoints de interfaz porque
facturan por hora aunque no se ejecute nada. La VPC, subredes, tabla de rutas,
gateway y cluster no tienen coste propio; sí lo tienen la **IPv4 pública** de
cada tarea mientras corre, el cómputo Fargate (0,5 vCPU / 1 GB, mínimo de un
minuto por tarea), los logs y la transferencia de datos. La conectividad real
no se ha probado: se verifica en la primera ejecución controlada.

**Publicar y activar (fase posterior, con aprobación)**

1. Publicar la imagen y anotar su **digest de manifiesto**.
2. Desplegar con `PipelineImageUri=<repositorio>@sha256:<digest>` (añadirlo a
   la lista completa de parámetros de "Alertas por correo"), manteniendo
   `PipelineEnabled=false`, y revisar el change set.
3. Primera ejecución controlada **manual** de una sesión, comprobando en el
   historial: que un exit code ≠ 0 se detecta (la documentación no aclara si
   `runTask.sync` falla el estado o termina con éxito; la definición cubre
   ambos casos), el formato de `Containers[0].ExitCode` en la salida, la
   respuesta de `listExecutions` (`Executions`) y que el ámbito IAM de
   `DescribeTasks`/`StopTask` basta.
4. Solo después, `PipelineEnabled=true`; el calendario sigue siendo una
   decisión aparte.

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
`f1-model-math.md`) se despliega en dos piezas, cada una creada **solo cuando su
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
sam deploy --config-env dev --parameter-overrides   ParameterKey=ProjectName,ParameterValue=f1-race-intelligence   ParameterKey=Environment,ParameterValue=dev   'ParameterKey=AvailabilitySchedule,ParameterValue="rate(15 minutes)"'   ParameterKey=AvailabilityScheduleState,ParameterValue=DISABLED   ParameterKey=PipelineEnabled,ParameterValue=false   ParameterKey=LogRetentionDays,ParameterValue=30   "ParameterKey=AlertEmail,ParameterValue=${ALERT_EMAIL}"   "ParameterKey=PipelineImageUri,ParameterValue=${PIPELINE_IMAGE}"   "ParameterKey=ForecastImageUri,ParameterValue=${FORECAST_IMAGE}"   ParameterKey=ModelScheduleState,ParameterValue=DISABLED   ParameterKey=LiveForecastScheduleState,ParameterValue=DISABLED
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
python -m f1_race_intelligence --stage m7-evaluate --split test   # solo para el informe final
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
- **Test.** `--split test` no promueve nunca y no debe usarse para decidir nada.
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
  15 min y `PipelineEnabled=false`, cada chequeo crea una ejecución STANDARD
  (~96/día, pocas transiciones): coste bajo pero no cero.
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

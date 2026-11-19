# Infraestructura AWS (SAM / CloudFormation)

> **Estado: NO desplegado.** Este directorio define la infraestructura base y la
> orquestación inicial. No crea datos, no ejecuta extracciones, no entrena ni
> publica modelos (sin M7, sin SageMaker). M2–M6 existen en la state machine
> solo como estados *placeholder*.

## Estructura

```
infra/
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
        PlanSessions                          (placeholder)
        PerSessionPipeline  Map, concurrencia 1
          M2_Extraction → M3_Validation → M4_Consolidation → M5_Features   (placeholders)
        M6_ModelingDataset                    (placeholder)
        Catch (solo en el Map) ─► NotifyPipelineFailure ─┘ ─► PipelineFailed

  S3 data lake <p>-<env>-<account>-<region>      ECR <p>-<env>-pipeline      CloudWatch Logs (retención parametrizada)
  raw/ validated/ consolidated/ features/        (imagen batch M2–M6)        /aws/lambda/…  /aws/vendedlogs/states/…
  modeling/ models/ manifests/                                               /aws/ecs/… (jobs futuros)
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
| Step Functions | STANDARD; logs `ALL` a CloudWatch; M2–M6 como `Pass` |
| SNS | topic de alertas críticas; política para CloudWatch de la misma cuenta; suscripción e-mail opcional |
| ECR | tags inmutables, escaneo al subir, expira sin tag a 7 d, conserva 30 imágenes; `RetainExceptOnCreate` |
| CloudWatch | 3 log groups con retención `LogRetentionDays`; 4 alarmas |

## IAM (mínimo privilegio)

| Rol | Confía en | Permisos |
|---|---|---|
| `…-availability-lambda` | `lambda.amazonaws.com` | escribir en su log group; `events:PutEvents` en el bus de orquestación; `sns:Publish` en el topic de alertas. **Sin S3.** |
| `…-pipeline-states` | `states.amazonaws.com` | `sns:Publish` en el topic; APIs de entrega de logs (`Resource: "*"` exigido por AWS). **Sin S3, sin ECS todavía.** |
| `…-availability-to-states` | `events.amazonaws.com` | `states:StartExecution` solo sobre la state machine |
| `…-data-processing` (futuro) | `ecs-tasks.amazonaws.com` | `s3:ListBucket` del bucket; `GetObject`/`PutObject` en `raw/ validated/ consolidated/ features/ modeling/ manifests/`. Sin `Delete*`, sin `models/`. |
| `…-data-processing-exec` (futuro) | `ecs-tasks.amazonaws.com` | pull de la imagen del repositorio ECR (`GetAuthorizationToken` requiere `*`); escribir en `/aws/ecs/…` |

Todas las relaciones de confianza llevan `aws:SourceAccount`, salvo la del rol
de la Lambda, que usa la confianza estándar de `lambda.amazonaws.com`: no se
pudo verificar que Lambda admita esa condición en roles de ejecución, y si no la
admite `CreateFunction` podría fallar. No hay políticas administradas de AWS ni
`AdministratorAccess`.

`PlanSessions` y `M6_ModelingDataset` son estados `Pass` y Step Functions no
admite `Retry`/`Catch` en ellos; hoy solo el `Map` captura errores. Al
convertirlos en `Task` de ECS hay que añadirles `Retry` y `Catch` hacia
`NotifyPipelineFailure`.

`s3:ListBucket` no se restringe por `s3:prefix` a propósito: sin ese permiso,
`HeadObject` sobre una clave inexistente devuelve 403 en vez de 404 y la
idempotencia del pipeline ("¿ya existe este archivo?") dejaría de funcionar.

## Parámetros

| Parámetro | Por defecto | Nota |
|---|---|---|
| `ProjectName` | `f1-race-intelligence` | prefijo de nombres y tag |
| `Environment` | `dev` | `dev` · `staging` · `prod` |
| `S3BucketName` | vacío | vacío ⇒ `<project>-<env>-<account>-<region>` |
| `AvailabilitySchedule` | `rate(15 minutes)` | solo valores ≥ 15 min |
| `AvailabilityScheduleState` | `DISABLED` | `samconfig.toml` también lo fija en `DISABLED` (dev y prod); activarlo es una decisión explícita |
| `PipelineEnabled` | `false` | dejar en `false` hasta desplegar los jobs |
| `LogRetentionDays` | `30` | valores válidos de CloudWatch |
| `NoncurrentVersionExpirationDays` | `90` | |
| `OpenF1BaseUrl` | `https://api.openf1.org/v1` | |
| `AlertEmail` | vacío | vacío ⇒ **sin suscripción: nadie recibe las alertas**; ver "Alertas por correo" |

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
REPO=$(aws cloudformation describe-stacks --stack-name f1-race-intelligence-dev \
  --query "Stacks[0].Outputs[?OutputKey=='PipelineImageRepositoryUri'].OutputValue" --output text)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build -t "$REPO:$(git rev-parse --short HEAD)" .
docker push "$REPO:$(git rev-parse --short HEAD)"
```

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
repositorio ECR del pronóstico y los roles):

| Pieza | Recurso | Se crea si | Qué hace |
|---|---|---|---|
| Entrenamiento / actualización | ECS Fargate (`RaceModelTaskDefinition`, imagen batch) + Step Functions `…-race-model` | `ModelImageTag`, `ModelVpcId` y `ModelSubnetIds` no vacíos | `{"action":"train"}`: `m7-train` → `m7-evaluate --promote`. `{"action":"update"}`: `m7-update` |
| Pronóstico | Lambda imagen `…-race-forecast` (`Dockerfile.forecast`) | `ForecastImageTag` no vacío | `{"session_key":…, "lap":…}` o `{"mode":"live"}` → `forecasts/m7/…` |

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

```bash
STACK=f1-race-intelligence-dev
TAG=$(git rev-parse --short HEAD)
out() { aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
BATCH_REPO=$(out PipelineImageRepositoryUri)
FORECAST_REPO=$(out ForecastImageRepositoryUri)
aws ecr get-login-password | docker login --username AWS --password-stdin "${BATCH_REPO%%/*}"
docker build --platform linux/amd64 -t "$BATCH_REPO:$TAG" . && docker push "$BATCH_REPO:$TAG"
docker build --platform linux/amd64 -f Dockerfile.forecast -t "$FORECAST_REPO:$TAG" . && docker push "$FORECAST_REPO:$TAG"

# VPC por defecto y sus subredes públicas (las tareas necesitan salida a S3/ECR/Logs)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" --query "Subnets[].SubnetId" --output text | tr '\t' ',')

: "${ALERT_EMAIL:?Define ALERT_EMAIL con la dirección real del destinatario}"
cd infra && sam build
sam deploy --config-env dev --parameter-overrides \
  ParameterKey=ProjectName,ParameterValue=f1-race-intelligence \
  ParameterKey=Environment,ParameterValue=dev \
  'ParameterKey=AvailabilitySchedule,ParameterValue="rate(15 minutes)"' \
  ParameterKey=AvailabilityScheduleState,ParameterValue=DISABLED \
  ParameterKey=PipelineEnabled,ParameterValue=false \
  ParameterKey=LogRetentionDays,ParameterValue=30 \
  "ParameterKey=AlertEmail,ParameterValue=${ALERT_EMAIL}" \
  "ParameterKey=ModelImageTag,ParameterValue=${TAG}" \
  "ParameterKey=ForecastImageTag,ParameterValue=${TAG}" \
  "ParameterKey=ModelVpcId,ParameterValue=${VPC}" \
  "ParameterKey=ModelSubnetIds,ParameterValue='${SUBNETS}'" \
  ParameterKey=ModelScheduleState,ParameterValue=DISABLED \
  ParameterKey=LiveForecastScheduleState,ParameterValue=DISABLED
```

En PowerShell, el mismo despliegue sigue el patrón protegido de "Alertas por
correo" añadiendo los seis parámetros `Model*`/`Forecast*`/`LiveForecast*`.

Primera carga del modelo y pruebas (los datos `raw/` deben estar en el bucket):

```bash
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

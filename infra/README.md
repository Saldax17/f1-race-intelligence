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
        Catch ─► NotifyPipelineFailure ─┘ ─► PipelineFailed

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
| S3 `DataBucket` | versionado; SSE-S3; bloqueo total de acceso público; `BucketOwnerEnforced`; política que niega tráfico sin TLS; lifecycle: aborta multipart a 7 d, expira versiones no actuales (parámetro, 90 d), limpia delete markers, `raw/` a Intelligent-Tiering a 30 d, `manifests/` a 90 d, `logs/` expira a 365 d; **nunca expira datasets actuales**; `DeletionPolicy: Retain` |
| Lambda `openf1-availability` | Python 3.12 arm64, 128 MB, 15 s; sin VPC; sin reintentos asíncronos; destinos OnSuccess→bus, OnFailure→SNS; métricas EMF `OpenF1Available`, `OpenF1LatencyMs` |
| EventBridge | regla programada (parámetros `AvailabilitySchedule`, `AvailabilityScheduleState`); bus propio + regla de resultados |
| Step Functions | STANDARD; logs `ALL` a CloudWatch; M2–M6 como `Pass` |
| SNS | topic de alertas críticas; política para CloudWatch de la misma cuenta; suscripción e-mail opcional |
| ECR | tags inmutables, escaneo al subir, expira sin tag a 7 d, conserva 30 imágenes; `Retain` |
| CloudWatch | 3 log groups con retención `LogRetentionDays`; 4 alarmas |

## IAM (mínimo privilegio)

| Rol | Confía en | Permisos |
|---|---|---|
| `…-availability-lambda` | `lambda.amazonaws.com` | escribir en su log group; `events:PutEvents` en el bus de orquestación; `sns:Publish` en el topic de alertas. **Sin S3.** |
| `…-pipeline-states` | `states.amazonaws.com` | `sns:Publish` en el topic; APIs de entrega de logs (`Resource: "*"` exigido por AWS). **Sin S3, sin ECS todavía.** |
| `…-availability-to-states` | `events.amazonaws.com` | `states:StartExecution` solo sobre la state machine |
| `…-data-processing` (futuro) | `ecs-tasks.amazonaws.com` | `s3:ListBucket` del bucket; `GetObject`/`PutObject` en `raw/ validated/ consolidated/ features/ modeling/ manifests/`. Sin `Delete*`, sin `models/`. |
| `…-data-processing-exec` (futuro) | `ecs-tasks.amazonaws.com` | pull de la imagen del repositorio ECR (`GetAuthorizationToken` requiere `*`); escribir en `/aws/ecs/…` |

Todas las relaciones de confianza llevan `aws:SourceAccount`. No hay políticas
administradas de AWS ni `AdministratorAccess`.

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
| `AvailabilityScheduleState` | `ENABLED` | apagar el chequeo sin borrar nada |
| `PipelineEnabled` | `false` | dejar en `false` hasta desplegar los jobs |
| `LogRetentionDays` | `30` | valores válidos de CloudWatch |
| `NoncurrentVersionExpirationDays` | `90` | |
| `OpenF1BaseUrl` | `https://api.openf1.org/v1` | |
| `AlertEmail` | vacío | requiere confirmar la suscripción |

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

# 1. Construir y desplegar dev (muestra el change set y pide confirmación)
cd infra
sam build
sam deploy --config-env dev
# opcional, alertas por correo:
sam deploy --config-env dev --parameter-overrides \
  ProjectName=f1-race-intelligence Environment=dev PipelineEnabled=false AlertEmail=tu-correo@ejemplo.com

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

## Eliminar

`sam delete --stack-name f1-race-intelligence-dev` borra el stack pero
**conserva** el bucket y el repositorio ECR (`DeletionPolicy: Retain`); se
eliminan a mano si realmente se quiere perder los datos.

## Riesgos conocidos

- Con el chequeo cada 15 min y `PipelineEnabled=false`, cada chequeo crea una
  ejecución STANDARD (~96/día, pocas transiciones): coste bajo pero no cero.
- La Lambda sale a internet sin VPC. Si más adelante se mete en una VPC hará
  falta NAT.
- El topic SNS no usa KMS: las alarmas de CloudWatch no pueden publicar en un
  topic cifrado con la clave administrada `aws/sns`; cifrarlo requiere una CMK
  con política para CloudWatch.
- `session_key=7953` es el objetivo del sondeo: si OpenF1 lo retirara, el
  chequeo daría 404 → `openf1_error`. Cambiarlo es editar
  `OPENF1_PROBE_PATH`.
- El rol de datos da `ListBucket` sobre todo el bucket (ver arriba).

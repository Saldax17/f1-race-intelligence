# Publicación inmutable de modelos M7

M7 empaqueta una versión ya registrada en MLflow para que M8 pueda recuperarla
y verificarla. La publicación solo lee el run seleccionado: no ajusta,
reentrena ni registra otro modelo. Se exige un URI de versión numérica exacta,
por ejemplo `models:/race-lap-time/3`; no se aceptan alias ni `latest`.

## Arquitectura y contenido

`modeling/model_package.py` descarga los artefactos del run asociado a la
versión registrada y comprueba su identidad, firma de entrada, reporte y
configuración M7, procedencia M6, `input_fingerprint` y hashes M6. La ruta
`PredictionService.load_published_package` valida el paquete antes de cargar el
modelo MLflow pyfunc y el preprocesador M6. También compara los hashes del
preprocesador y del esquema con `verified_artifact_hashes` de la provenance
M7. El flujo M7→MLflow→M8 existente continúa disponible sin cambios.

La estructura lógica es:

```text
published-models/<nombre-registrado>/versions/<versión-numérica>/
├── model/                          paquete MLflow pyfunc completo
├── m6/
│   ├── preprocessor.joblib          preprocesador M6 exacto
│   ├── feature_schema.json          esquema de features M6 exacto
│   └── modeling_report_*.json       reporte M6 adjuntado por M7
├── data_provenance.json             procedencia M6/M7 y hashes M6
├── training_config.json             configuración M7
├── training_report.json             run, modelo, target, métricas y commit
├── manifest.json                    contrato y SHA-256 de cada artefacto
└── PUBLISHED                        sha256:<hash del manifiesto>
```

El manifiesto registra nombre, versión y URI del modelo, IDs de runs M6/M7,
firma de entrada ordenada, target, `input_fingerprint`, versión de datos,
commits disponibles y tamaño y SHA-256 de cada archivo. El manifiesto se
serializa como JSON canónico UTF-8, con claves ordenadas y separadores
compactos. `PUBLISHED` contiene el hash del manifiesto; así se verifica el
manifiesto sin incluir su propio hash recursivamente.

## Publicación, concurrencia y reintentos

Todos los artefactos y el manifiesto se crean mediante
`StorageBackend.write_bytes_if_absent`. El publicador los vuelve a leer y
verifica antes de crear `PUBLISHED` con otra escritura condicional, como último
paso. El lector solo acepta paquetes con marcador válido, conjunto exacto de
objetos y hashes correctos. Los prefijos parciales no son paquetes publicados.

En local, la creación condicional evita reemplazar un archivo existente. El
backend S3 usa `If-None-Match: *`. Si dos publicadores compiten, un objeto
preexistente con bytes idénticos se puede reutilizar; si los bytes difieren, el
publicador falla y no sobrescribe el objeto. Un lector concurrente no acepta
el paquete mientras no exista el marcador.

Un **409** o **412** durante una escritura condicional no autoriza a reemplazar
el objeto. Si el objeto está visible y sus bytes coinciden, se puede continuar;
si difieren, se rechaza la publicación. Si un 409 ocurre antes de que el objeto
sea visible, se informa el conflicto y el lector rechaza el prefijo.

Si un PUT de `PUBLISHED` se completa pero el cliente recibe un timeout, el
reintento vuelve a calcular el manifiesto determinista y verifica el paquete
existente. Si el manifiesto coincide exactamente, devuelve el paquete
verificado sin escribir objetos; si el contenido difiere o el paquete no es
válido, falla explícitamente. Un fallo parcial sin marcador puede reanudarse
solo cuando los objetos que ya existen coinciden con la publicación solicitada.
Un objeto distinto impide completar el paquete.

## Recuperación operativa de prefijos inconsistentes

La publicación no elimina ni reescribe automáticamente objetos o marcadores.
Ante un error:

1. Identificar el nombre y la versión exactos y listar el prefijo
   `published-models/<nombre>/versions/<versión>`. Usar
   `read_model_package(backend, prefix)` para validar el marcador, manifiesto,
   pertenencia de objetos y hashes.
2. Retirar esa versión del uso de M8 hasta cerrar la investigación. No
   configurar el servicio para que elija `latest` ni para que use un paquete
   no verificado.
3. Conservar el prefijo para diagnóstico. No borrar objetos, no sustituir el
   manifiesto ni regenerar el marcador bajo el mismo identificador.
4. Si el prefijo no tiene marcador y sus objetos coinciden con el origen M7
   exacto, se puede reintentar la misma publicación. Si contiene objetos
   adicionales, una colisión con bytes distintos o un marcador inválido, no
   intentar repararlo in situ.
5. Registrar una nueva versión numérica en MLflow y publicarla en su nuevo
   prefijo. Validarla antes de cambiar la configuración de M8. Mantener el
   prefijo anterior fuera de servicio para auditoría.

`PackageAlreadyPublished` identifica una versión válida ya publicada con
contenido diferente; `InvalidPublishedPackage` identifica un marcador
`PUBLISHED` cuyo paquete no supera la verificación. La ausencia de marcador se
informa como paquete aún no publicado. Las colisiones o archivos adicionales
en un prefijo sin marcador se informan como inconsistencias y no se aceptan.

## Uso

Instalar las dependencias fijadas por el proyecto; esta funcionalidad no añade
dependencias:

```powershell
python -m pip install -r requirements/runtime-lock.txt
python -m pip install -r requirements/training-lock.txt
python -m pip install -r requirements/dev-lock.txt
python -m pip install -e ".[dev,training,serve]" --no-deps
```

Ejemplo de publicación local:

```python
from f1_race_intelligence.modeling.model_package import publish_model
from f1_race_intelligence.storage.backends.local import LocalStorageBackend

backend = LocalStorageBackend("model-store")
package = publish_model("models:/race-lap-time/3", backend)
```

Carga para M8:

```python
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import PredictionService

service = PredictionService.load_published_package(
    backend=backend,
    prefix=package.prefix,
    loader=MlflowPyfuncLoader(),
)
```

El backend S3 se entrega explícitamente como `S3StorageBackend`; en pruebas se
usa `FakeS3Client`. No se deben proporcionar credenciales reales ni realizar
publicaciones reales durante la validación local.

## Límites de seguridad y garantías

Las pruebas con `FakeS3Client` comprueban el tratamiento de respuestas 409/412,
fallos de red, resultados ambiguos y escrituras concurrentes dentro del
simulador. **No demuestran el comportamiento operacional de AWS S3 real** ni se
han realizado operaciones contra AWS.

Las escrituras condicionales protegen frente a sobrescrituras hechas mediante
esta aplicación. El hash del manifiesto detecta inconsistencias, pero no es
una firma criptográfica: una entidad con permisos para cambiar objetos,
manifiesto y marcador podría recalcularlos. Un despliegue real debe restringir
con IAM la escritura y eliminación, limitar quién puede publicar, conservar
auditoría y evaluar versionado/Object Lock y políticas que exijan escrituras
condicionales. La configuración exacta de bucket, IAM y retención queda fuera
de esta implementación y requiere validación operativa independiente.

## Pruebas

Las pruebas generan artefactos M6 y runs M7 sintéticos, sin OpenF1 ni datos
reales. La suite focalizada cubre equivalencia local/S3 simulado, publicación
concurrente de contenido distinto, lecturas durante la publicación, 409/412,
fallos parciales y reintentos, timeout ambiguo del marcador, integridad,
trazabilidad M6–M7, carga M8 y ausencia de `fit` durante inferencia.

```powershell
python -m pytest tests\unit\test_m7_m8_integration.py -q
python -m pytest -m "not integration"
```

Los resultados actualizados de cada ejecución se conservan en el informe de
validación de esta entrega: **34 pruebas focalizadas aprobadas** y **785
pruebas no integradas aprobadas, 2 excluidas**. El resultado completo incluyó
42 advertencias no bloqueantes de dependencias y cálculos ya existentes. Las
métricas de los fixtures sintéticos no son resultados académicos.

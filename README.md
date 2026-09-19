# RAG Lab

Proyecto educativo para comprender cada pieza de RAG y registrar qué documentos,
código, configuración y modelos produjeron un índice o una respuesta.

**Estado: cuarta iteración.** Funcionan ingestion, chunking manual, embeddings
reales locales, índice vectorial JSON y retrieval con cosine manual, búsqueda
exhaustiva, ranking y Top-K. Todavía no hay armado de contexto, llamadas a LLM
ni métricas. Las carpetas vacías señalan las etapas
que construiremos después.

```text
Documents → Ingestion → Chunking → Embeddings → Vector Index
          → Retrieval → Context Assembly → LLM → Answer → Evaluation
```

## Estructura

```text
data/raw/                 biblioteca + escenario original de migración
src/rag_lab/
  config.py               dataclasses y carga de JSON
  identity.py             SHA-256 y JSON canónico
  ingestion/              Document, lectura UTF-8 y corpus ID
  chunking/               Chunk y ventanas de caracteres
  dataset.py              composición e identidad del dataset
  __main__.py             CLI con argparse
  embeddings/             contrato y E5 local con ONNX Runtime
  indexing/               colección de vectores y persistencia
  retrieval/              query, cosine, ranking, Top-K y trazabilidad
  context/                pendiente: armado de contexto y prompt
  generation/             pendiente
  evaluation/             pendiente
config/pipeline.json      receta inicial
config/embedding.json     modelo, revisión, hashes y procesamiento fijados
config/retrieval.json     versión, métrica, algoritmo y Top-K
scripts/download_model.py descarga explícita y verificación de assets
.models/                 modelo local (ignorado por Git)
artifacts/indexes/        índices derivados JSON + checksum (ignorados por Git)
artifacts/manifests/      futuras ejecuciones (ignoradas por Git)
docs/manifest.md          contrato previsto de trazabilidad
docs/manifest.example.json plantilla, no una ejecución real
docs/embeddings-index.md   decisiones y formato implementado
docs/retrieval.md          explicación, CLI y trazabilidad de búsqueda
requirements-embeddings.lock.txt versiones instaladas para embeddings
tests/                    tests offline con test double explícito
```

Usamos `src/rag_lab` para tener un único paquete importable, sin clases base ni
frameworks. Separa explícitamente `context` de generación para enseñar el armado
del prompt. El nombre de la carpeta del workspace puede seguir siendo `rag`.

## Entorno local (PowerShell)

Para empezar desde GitHub:

```powershell
git clone https://github.com/YamiCueto/rag-example.git
cd rag-example
```

El repositorio distribuye código, configuración y corpus ficticio original. No
incluye entorno virtual, modelos descargados ni índices generados. En un clon nuevo,
crea el entorno con los comandos siguientes y ejecuta después la preparación inicial
del modelo e índice indicada en «Demostración y comparación». No se necesitan API keys.
Los archivos del corpus conservan sus bytes al pasar por Git para mantener sus hashes.

Python detectado durante esta iteración: **3.14.6**. El proyecto declara Python
**>=3.11**, pero únicamente se ha probado con la versión detectada.

```powershell
python -m venv .venv
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m rag_lab data/raw --preview-chunks 3
```

El entorno `.venv` se creó en la primera iteración. Ingestion, chunking y los tests
offline usan la biblioteca estándar. Para embeddings reales se instalaron
**ONNX Runtime 1.30.0** (inferencia CPU), **tokenizers 0.23.2** (tokenización del
modelo) y **NumPy 2.5.3** (tensores, pooling y normalización). Se verificaron wheels
Windows x64 compatibles con Python 3.14.6 y se ejecutó inferencia real en este equipo.
El lock registra también dependencias transitivas; no hay Torch ni frameworks RAG.
`setuptools>=68` es únicamente backend de empaquetado; no se validó el build del paquete.

## Configuración y versiones

`PipelineConfig` agrupa `PipelineVersions`, `ChunkingConfig` y dos `ModelConfig`.
Las dataclasses son inmutables y validan valores básicos al construirse.
`load_config` lee JSON, rechaza campos desconocidos y aplica defaults a los omitidos.
`to_dict()` produce la copia serializable que usará el manifest.

Las nueve capas tienen etiquetas iniciales `"1"`; no significan que todas estén
implementadas. `rag_release = "0.1.0"` identifica esta receta. El chunking usa
ventanas de **500 caracteres** y **50 caracteres** de solapamiento; caracteres de
Python (puntos de código Unicode), no tokens, bytes ni necesariamente caracteres
visuales completos. Se valida `chunk_size > 0` y `0 <= overlap < chunk_size`.
Para conservar compatibilidad, el campo Python y el JSON serializado se llaman
`overlap`. Al cargar JSON también se acepta `chunk_overlap`; usar ambos nombres
en un mismo objeto se rechaza. La CLI ofrece `--chunk-overlap`.

El modelo real se configura en `config/embedding.json`, separado de la receta
base para mantener la ejecución de ingestion/chunking sin dependencias opcionales.
`PipelineConfig.embedding` conserva sus defaults `null`; si se especifican modelo
o revisión allí, la CLI comprueba que coincidan con la configuración real.
Generación y evaluación siguen pendientes. `RetrievalConfig`, en
`config/retrieval.json`, contiene su propia versión efectiva y configuración;
la operación `search` no carga `pipeline.json`.

El índice es un **artefacto derivado**, no la fuente de verdad. Su identidad depende
del dataset de chunks y de la receta de embeddings/indexación. El
[manifest completo](docs/manifest.md) sigue siendo un diseño futuro; el artefacto
actual sí contiene dependencias, metadatos y vectores reales.

## Ingestion: del archivo al documento

Ingestion descubre recursivamente archivos `.txt` (también `.TXT`) y los carga
como UTF-8 estricto: un archivo inválido hace fallar la ejecución, sin reemplazar
caracteres silenciosamente. Otros tipos se ignoran. `Document` conserva:

- `document_id`, ruta relativa `source_path` y `source_sha256` de los bytes originales;
- `source_text`, texto fuente intacto, y `text`, texto normalizado;
- `encoding = "utf-8"` y `normalization = "newlines_to_lf_v1"`.

La única normalización convierte `\r\n` y `\r` en `\n`. No se recortan espacios,
no se altera mayúsculas/minúsculas y no se elimina BOM. Los offsets de chunks se
refieren al **texto normalizado**, no a posiciones de bytes del archivo. Dos
archivos con distintos saltos de línea pueden generar el mismo texto normalizado,
pero conservan distintos hashes e identidades de fuente.

## Chunking: ventanas visibles

Chunking divide el texto en fragmentos para las etapas posteriores. Empezamos con
caracteres porque las posiciones y el solapamiento son fáciles de inspeccionar.
Es una decisión pedagógica aprobada, no una recomendación universal. Después
podremos comparar caracteres, tokens y estrategias estructurales/semánticas.

El algoritmo comienza en `start = 0`, toma `text[start:end]`, con
`end = min(start + chunk_size, len(text))`, y avanza `chunk_size - overlap`.
Cuando llega al final, se detiene sin emitir un fragmento redundante de overlap.
Un documento vacío produce cero chunks; uno pequeño produce un solo chunk.

```text
Texto normalizado de 1050 caracteres, tamaño 500, overlap 50:
chunk-000: [  0,  500)  500 caracteres
chunk-001: [450,  950)  500 caracteres
chunk-002: [900, 1050)  150 caracteres
```

El inicio se incluye y el final se excluye. Los caracteres `[450, 500)` aparecen
tanto al final del primer chunk como al inicio del segundo: eso es overlap.
Cada `Chunk` conserva documento, ID, índice desde cero, offsets, texto y SHA-256
de los bytes UTF-8 de su texto. El fixture largo de tests verifica este ejemplo;
`biblioteca.txt` tiene menos de 500 caracteres y produce naturalmente un solo chunk.
Se agregó `migracion_pedidos.txt`, original para este laboratorio, que sí permite
ver ventanas `[0,500)`, `[450,950)`, `[900,1400)` y siguientes. Biblioteca se conserva.

## Identidades: fuente frente a representación

Todos los IDs usan SHA-256 sobre JSON canónico UTF-8: claves ordenadas, sin espacios
superfluos, Unicode sin escapar y sin NaN/Infinity. El objeto incluye `kind`,
`schema_version = 1` y `payload`; el ID se presenta como `<kind>-<sha256>`.
Los checksums de texto/archivo, en cambio, se calculan directamente sobre bytes.

| Identidad | Datos que entran en el hash |
| --- | --- |
| Documento | Ruta relativa POSIX y SHA-256 de los bytes fuente |
| Corpus | Lista de rutas relativas y hashes fuente, ordenada por ruta |
| Chunk | Documento, versión ingestion, normalización, versión/config chunking, índice, offsets y checksum del fragmento |
| Dataset de chunks | Corpus ID, versión ingestion, normalizaciones usadas, versión/config chunking y lista ordenada de chunk IDs |
| Índice vectorial | Dataset ID, provider/model/revision/dimensions, settings efectivos del embedding, versiones de embedding/index y formato JSON |

Mover la carpeta completa conserva las identidades: no participan rutas absolutas
ni timestamps. Renombrar un archivo dentro del corpus sí cambia su identidad y la
del corpus, aunque sus bytes no cambien. No se admiten rutas de documento duplicadas.
Un corpus vacío tiene una identidad definida y produce cero chunks.

El dataset ordena documentos por ruta y chunks por posición. Cambiar `chunk_size`
o `overlap` cambia su identidad incluso cuando un documento pequeño produce el
mismo texto. El corpus mantiene su identidad. Las etiquetas `corpus_version` y
`rag_release` no sustituyen al contenido ni afectan esta identidad; versiones y
parámetros de embeddings, índice, retrieval, prompt, generación y evaluación
tampoco intervienen porque son etapas posteriores.

```text
Corpus v1                     Corpus v1
    ↓                             ↓
Ingestion v1                  Ingestion v1
    ↓                             ↓
Chunking v1 (500/50)           Chunking v2 (1000/100)
    ↓                             ↓
Chunk Dataset A               Chunk Dataset B
```

Mismo corpus, distinta representación para retrieval. Cambiar solo los parámetros
ya modifica la identidad del dataset, incluso sin cambiar la etiqueta de versión.
El índice depende de esa representación: cambiar el dataset cambia su identidad
y exige construir los vectores derivados correspondientes.

Las versiones ingestion/chunking son etiquetas explícitas del código; al cambiar
su comportamiento deben actualizarse. Los IDs actuales incluyen también los
resultados reales (chunk IDs/checksums). Esto permite comparar datos; no sustituye
el futuro snapshot de código y entorno del manifest completo.

## Demostración y comparación

Desde la raíz del proyecto, después de definir `PYTHONPATH` como arriba:

```powershell
.\.venv\Scripts\python.exe -m rag_lab data/raw --preview-chunks 3
```

Para un experimento posterior, el mismo comando acepta
`--chunk-size 1000 --chunk-overlap 100`. Los overrides solo se aplican a esa ejecución;
no modifican `config/pipeline.json`. `--config` permite cargar otra receta y
`--preview-chunks 0` omite vistas previas. La CLI muestra corpus ID, dataset ID,
documentos, parámetros, total de chunks y metadatos para inspección.

Sin `--build-index`, la CLI conserva la demo en memoria de ingestion/chunking.
En un clon nuevo, ejecuta una vez estos comandos para descargar el modelo verificado
y construir tu índice local. Si ya tienes el índice, no necesitas repetirlos para buscar:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-embeddings.lock.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe -m rag_lab data/raw --build-index --preview-chunks 4
```

La instalación y descarga inicial requieren red; solo descargan paquetes y pesos
públicos. La inferencia posterior es local y no envía documentos. La descarga
verifica los dos SHA-256 fijados, y la CLI vuelve a verificarlos antes de inferir.
El modelo y tokenizer suman aproximadamente **135 MB**. `--embedding-config`,
`--model-directory` y `--artifacts` permiten indicar sus rutas.

## Del texto al vector

```text
Text → Embedding Model → Vector de 384 números

Corpus → Chunk Dataset → Embedding Model → Vector Index
```

El modelo seleccionado es **intfloat/multilingual-e5-small**, multilingüe, ejecutado
en CPU mediante su export ONNX cuantizado. La receta añade `passage: ` al texto,
tokeniza, ejecuta el modelo, promedia representaciones de tokens usando la máscara
y normaliza el vector. Estas operaciones están visibles en `onnx_local.py`.
El límite es 512 tokens; si se supera, se rechaza el chunk sin truncarlo.

Un embedding **no guarda significado como texto legible**. Sus dimensiones
individuales normalmente no tienen interpretación humana directa; la representación
numérica aprendida permite comparar la consulta con los chunks en retrieval.
Cambiar de modelo o revisión normalmente cambia ese espacio numérico y obliga
a reconstruir los vectores. No se mezclan modelos en un mismo índice.

| Cambio | Corpus ID | Chunk dataset ID | Vector index ID |
| --- | --- | --- | --- |
| Contenido fuente | Cambia | Cambia | Cambia |
| Chunking 500/50 → 1000/100 | Igual | Cambia | Cambia |
| Modelo/revisión de embedding | Igual | Igual | Cambia |

`Vector Index = artefacto derivado`. `Vector Index ID ≠ Artifact SHA-256`:
el primero identifica la receta/dependencias; el segundo identifica los bytes
escritos, incluidos vectores y procedencia de runtime. Dos ejecuciones pueden
compartir receta y diferir en bytes por entorno o aritmética numérica. El sistema
rechaza sobrescribir la misma identidad con contenido distinto; tampoco sobrescribe
un artefacto corrupto. Un resultado idéntico verificado se reutiliza.

Los tests de identidad usan **DeterministicTestEmbedder**, ubicado exclusivamente
en `tests/`: es un **TEST DOUBLE NO SEMÁNTICO**, no sustituto del modelo real.
Consulta [embeddings e índice](docs/embeddings-index.md) para el formato, fuentes,
compatibilidad y límites de reproducibilidad.

## Búsqueda sobre el índice existente

Con el entorno y modelo ya disponibles, la operación `search` carga y valida el
índice, genera únicamente el embedding de la consulta y compara todos los vectores:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m rag_lab search --query "¿Cómo se evita descontar inventario dos veces cuando llega el mismo evento duplicado?" --top-k 3
```

Este comando selecciona el único índice local. Si tienes varios, indica `--index`
con la ruta `index.json` que imprimió su construcción.
No reconstruye el corpus ni el índice. E5 recibe `query: ` para la consulta y
`passage: ` para los documentos. La CLI muestra todos los scores, ranking completo,
Top-K e identidades; las vistas previas de texto se limitan a 240 caracteres,
mientras `SearchResult.text` conserva el chunk completo.

Top-K selecciona los primeros K del ranking: no filtra por calidad ni garantiza
evidencia relevante. La [guía de retrieval](docs/retrieval.md) explica la fórmula,
complejidad, compatibilidad de modelos y separación entre similitud y relevancia.
Los tests offline validan operaciones con vectores controlados y dobles; una
búsqueda real valida además el flujo local E5, sin constituir evaluación RAG.

En el cierre de la cuarta iteración pasaron **51/51 tests** y se ejecutó una
única búsqueda oficial con E5 sobre los seis vectores existentes. El primer
resultado fue el chunk 2 de `migracion_pedidos.txt`, con score `0.888977564`:
contiene el mecanismo de idempotencia basado en la clave del evento. Esto verifica
el flujo y permite inspeccionar evidencia; no mide calidad sobre un dataset.

**STOP:** revisar la cuarta iteración antes de autorizar la siguiente fase.

# Contrato previsto del manifest (schema 1)

Este documento diseña el formato. `manifest.example.json` es una plantilla
`planned`, no evidencia de una ejecución ni de un índice existente. Todavía no
hay generador ni validador del manifest completo. La tercera iteración sí genera
un índice JSON con `vector_index_id`, `build_spec`, vectores y checksum de archivo;
su formato está en [embeddings-index.md](embeddings-index.md). No rellena esta
plantilla de ejecución completa. `null` significa pendiente o no
aplicable; nunca un valor medido. Los manifests reales irán en
`artifacts/manifests/<run_id>.json`.

## Tres identidades diferentes

- `schema_version`: versión del formato del manifest.
- `rag_release` y las versiones de cada capa: etiquetas humanas de una receta.
- `build_spec_sha256`: identidad derivada de las dependencias efectivas del índice.
  `artifact_sha256` identifica los bytes del resultado; no es el mismo hash.

Cada ejecución futura registrará `run_id` único, fecha UTC, operación
(`build_index`, `query` o `evaluate`), estado (`completed` o `failed`) y una copia
completa de la configuración efectiva en `config_snapshot`. En un fallo se
añadirá `error` con etapa y mensaje, sin afirmar que hay un artefacto completo.

`provenance` registrará versión exacta de Python, dependencias resueltas y revisión
del código. Si no hay commit o el árbol está modificado, se conservará un snapshot
del código con su checksum; una revisión Git sola no reconstruye cambios locales.
Guardar hashes exige también conservar los archivos que esos hashes identifican.

## Identidad del índice derivado

El futuro `build_spec` contendrá:

1. Corpus: lista ordenada de rutas relativas y SHA-256 de los bytes de cada archivo.
2. Ingestion: versión, configuración efectiva y hash del código relevante.
3. Chunking: versión, estrategia, unidad, tamaño, overlap y hash del código.
4. Embeddings: proveedor/modelo, revisión resuelta, dimensión, normalización,
   parámetros efectivos, código y dependencias relevantes.
5. Indexación: formato/algoritmo, versión, configuración y dependencias relevantes.

La identidad del dataset de chunks implementada en la segunda iteración puede
referenciarse como dependencia adicional del índice. Depende del corpus real,
ingestion, normalización, configuración/versionado de chunking y chunks obtenidos.
La tercera iteración implementa `vector_index_id` como hash tipado del `build_spec`
actual; depende transitivamente del corpus mediante `chunk_dataset_id`. El snapshot
automático de código y la trazabilidad completa de cada ejecución siguen pendientes.

Se serializará como JSON UTF-8 con claves ordenadas, sin espacios superfluos y
sin valores NaN/Infinity, y se calculará SHA-256. Los archivos del corpus se
ordenarán por ruta relativa con separador `/`; sus bytes no se normalizarán.
Fechas, rutas absolutas y run IDs quedan fuera de la identidad.

La implementación actual guarda `artifacts/indexes/<vector_index_id>/index.json`
y `index.sha256`. Publica ambos archivos mediante un directorio temporal y un
renombrado en el mismo filesystem. Si la ruta existe, verifica checksum y bytes
antes de reutilizarla; ante diferencias falla sin sobrescribir. El futuro manifest
podrá referenciar ese ID y checksum sin confundirlos.

Cambiar corpus, ingestion, chunking, embeddings o indexación invalida la identidad
del índice. Cambiar solo retrieval, prompt, generación o dataset de evaluación
produce una nueva receta/ejecución, pero no obliga a reconstruir ese índice.

## Respuestas y evaluación futuras

`query` registrará pregunta, identidad/checksum del índice usado, configuración
de retrieval, IDs y scores de chunks recuperados, orden y texto del contexto,
versión y texto del prompt, modelo/revisión/configuración efectiva de generación
y respuesta. `evaluation` registrará versión y checksum del dataset, preguntas,
referencias y resultados por pregunta, más versión de las métricas.

Para comparar 500/50 frente a 1000/100 se conservará el mismo corpus y dataset,
cambiará la configuración de chunking y se construirán dos índices con identidades
distintas. El registro permite auditar y repetir el procedimiento; no garantiza
igualdad numérica de modelos remotos o hardware no determinista. La política de
retención y privacidad de preguntas/contexto se decidirá antes de usar datos reales.

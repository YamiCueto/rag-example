# Catálogo de Arquitectura Visual del Laboratorio RAG (Archify)

Esta carpeta contiene la documentación visual, navegable y verificable de la arquitectura real del laboratorio RAG, generada con **Archify**.

Cada diagrama ha sido validado bajo el perfil `showcase` de Archify (sin colisiones, 0 errores, 0 advertencias) y verificado visualmente en resoluciones de escritorio estándar (1440×900) y alta densidad (2048×1320).

---

## Índice de Diagramas

| Diagrama | Tipo Archify | HTML Interactivo (Showcase) | Imagen Estática (GitHub Preview) | Especificación JSON |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pipeline General** | `dataflow` | [pipeline.html](pipeline.html) | [pipeline.png](pipeline.png) | [pipeline.dataflow.json](pipeline.dataflow.json) |
| **2. Flujo de Consulta** | `workflow` | [query-flow.html](query-flow.html) | [query-flow.png](query-flow.png) | [query-flow.workflow.json](query-flow.workflow.json) |
| **3. Identidades y Versionamiento** | `dataflow` | [identities.html](identities.html) | [identities.png](identities.png) | [identities.dataflow.json](identities.dataflow.json) |

---

## Cómo Visualizar e Interactuar con los Diagramas

### Visualización rápida en GitHub
GitHub no ejecuta JavaScript interactivo dentro de archivos HTML por políticas de seguridad (CSP). Si estás navegando en GitHub:
1. Las imágenes estáticas **PNG** (`*.png`) y los bloques Mermaid del `README.md` principal te ofrecen una vista previa instantánea.
2. Para la experiencia completa, clona el repositorio y abre los archivos HTML localmente.

### Experiencia Interactiva Completa (Local)
Los archivos HTML son **autocontenidos** (no requieren conexión a internet ni servidor web). Puedes abrirlos directamente con cualquier navegador moderno:

```powershell
# En Windows PowerShell desde la raíz del proyecto:
Start-Process "docs/architecture/pipeline.html"
Start-Process "docs/architecture/query-flow.html"
Start-Process "docs/architecture/identities.html"
```

### Controles y Herramientas en el Visor Archify:
- **Reproductor de Historias (Story Mode)**: Presiona el botón de reproducción `▶` o la barra espaciadora para ver un recorrido guiado paso a paso por la arquitectura, con anotaciones explicativas.
- **Selector de Vistas**: Cambia entre la vista completa (`All`), la vista de componentes implementados y la vista de límites del sistema.
- **Navegación Vectorial**: Rueda del ratón para Zoom infinito, arrastrar para paneo, o botón de centrado para ajustar a la pantalla.
- **Mini-Mapa**: Navegador de cuadrante inferior para situarte en diagramas extensos.
- **Toggle de Tema**: Cambia entre modo oscuro (Dark Mode) y modo claro (Light Mode) según tu preferencia.
- **Inspección de Contratos**: Haz clic en cualquier nodo o conector para ver su metadata, inputs/outputs, complejidad algorítmica y detalles de implementación.

---

## Resumen de los Diagramas

### 1. Pipeline General (`pipeline.html`)
- **Propósito**: Mapear de extremo a extremo el flujo de datos desde los archivos crudos de texto en disco (`data/raw/*.txt`) hasta la generación del índice vectorial persistente y la búsqueda semántica.
- **Límites claros**: Distingue visualmente los componentes completados (Ingesta, Chunking, Dataset, Embeddings locales ONNX, Índice JSON, Retrieval exhaustivo) de las fases pendientes (Armado de contexto, LLM, Evaluación).

### 2. Flujo de Consulta y Retrieval (`query-flow.html`)
- **Propósito**: Detallar qué ocurre exactamente cuando un usuario formula una pregunta.
- **Estructura en Lanes**:
  - *Lane 1: Entrada y Prefijo*: Validación de pregunta, agregado de `query: `.
  - *Lane 2: Inferencia Local ONNX*: Tokenización, ONNX Runtime CPU, pooling y normalización L2.
  - *Lane 3: Comparación, Ranking y Top-K*: Cálculo exhaustivo de similitud coseno $O(N \times D)$, ordenamiento descendente determinista $O(N \log N)$ y corte Top-K.

### 3. Cadena de Identidades y Versionamiento (`identities.html`)
- **Propósito**: Demostrar la trazabilidad determinista de punta a punta del laboratorio.
- **Contratos visualizados**:
  - `Document ID` y `Corpus ID` basados en SHA-256 de archivos fuente.
  - `Chunk ID` y `Chunk Dataset ID` basados en configuración de chunking y offsets.
  - **Diferencia crítica**: `Vector Index ID` (identidad lógica derivada de la receta declarativa `build_spec`) vs. `Artifact Checksum` (integridad física de los bytes reales de `index.json` en disco).
  - `Retrieval Experiment ID` que enlaza la consulta, la receta lógica, el artefacto físico y el Top-K.


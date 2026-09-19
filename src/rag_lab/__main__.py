"""Construcción del índice y búsqueda exhaustiva local, sin generación."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from rag_lab.config import load_config
from rag_lab.dataset import build_chunk_dataset
from rag_lab.ingestion import load_corpus


def main() -> None:
    if sys.argv[1:2] == ["search"]:
        search_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Inspecciona ingestion y chunking de archivos .txt")
    parser.add_argument("corpus", nargs="?", default="data/raw")
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--preview-chunks", type=int, default=3)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--embedding-config", default="config/embedding.json")
    parser.add_argument("--model-directory", default=".models/multilingual-e5-small")
    parser.add_argument("--artifacts", default="artifacts/indexes")
    args = parser.parse_args()
    if args.preview_chunks < 0:
        parser.error("--preview-chunks debe ser >= 0")
    try:
        config = load_config(args.config)
        overrides = {}
        if args.chunk_size is not None:
            overrides["chunk_size"] = args.chunk_size
        if args.chunk_overlap is not None:
            overrides["overlap"] = args.chunk_overlap
        config = replace(config, chunking=replace(config.chunking, **overrides))
        documents = load_corpus(args.corpus)
        dataset = build_chunk_dataset(documents, config)
        index = artifact = None
        if args.build_index:
            from rag_lab.embeddings import load_embedding_spec
            from rag_lab.embeddings.onnx_local import LocalOnnxEmbedder
            from rag_lab.indexing import build_vector_index, persist_index

            spec = load_embedding_spec(args.embedding_config)
            if config.embedding.model_id not in (None, spec.model) or config.embedding.revision not in (None, spec.revision):
                raise ValueError("La receta pipeline y embedding-config indican modelos o revisiones diferentes")
            embedder = LocalOnnxEmbedder(spec, args.model_directory)
            index = build_vector_index(dataset, embedder, config.versions.embedding, config.versions.index)
            artifact = persist_index(index, args.artifacts)
    except ImportError as error:
        parser.error(f"Instala requirements-embeddings.lock.txt para construir el índice: {error}")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    print(f"Corpus: {args.corpus}")
    print(f"Documents: {len(documents)}")
    print(f"Chunk strategy: {config.chunking.strategy} (Python characters)")
    print(f"Chunk size: {config.chunking.chunk_size}")
    print(f"Overlap: {config.chunking.overlap}")
    print(f"Total chunks: {len(dataset.chunks)}")
    print(f"Corpus ID: {dataset.corpus_id}")
    print(f"Chunk dataset ID: {dataset.chunk_dataset_id}")
    for document in documents:
        print(f"\nDocument: {document.source_path}")
        print(f"Document ID: {document.document_id}")
        print(f"Source SHA-256: {document.source_sha256}")
    for chunk in dataset.chunks[:args.preview_chunks]:
        print(f"\nchunk-{chunk.chunk_index:03d} chars [{chunk.start_char}, {chunk.end_char})")
        print(f"Document ID: {chunk.document_id}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"SHA-256: {chunk.text_sha256}")
        print(f"Preview: {json.dumps(chunk.text[:100], ensure_ascii=True)}")

    if index is not None:
        embedding = index.build_spec["embedding"]
        print("\nEmbedding:")
        for key in ("provider", "model", "revision", "dimensions"):
            print(f"  {key}: {embedding[key]}")
        print(f"Vector Index ID: {index.vector_index_id}")
        print(f"Artifact: {artifact.path.as_posix()}")
        print(f"Artifact SHA-256: {artifact.sha256}")
        print(f"Reused verified artifact: {artifact.reused}")
        if index.records:
            example = next((record for record in index.records if record["metadata"]["end_char"] == 500), index.records[0])
            print(f"\nExample chunk: {json.dumps(example['metadata']['text'][:140], ensure_ascii=True)}")
            print(f"Embedding preview: {[round(value, 6) for value in example['vector'][:6]]} ...")
            print(f"Dimensions: {example['embedding']['dimensions']}")


def search_main(arguments: list[str]) -> None:
    from rag_lab.embeddings import load_embedding_spec
    from rag_lab.embeddings.onnx_local import LocalOnnxEmbedder
    from rag_lab.indexing import load_index
    from rag_lab.retrieval import load_retrieval_config, search

    parser = argparse.ArgumentParser(description="Query → cosine exhaustivo → ranking → Top-K")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--retrieval-config", default="config/retrieval.json")
    parser.add_argument("--embedding-config", default="config/embedding.json")
    parser.add_argument("--model-directory", default=".models/multilingual-e5-small")
    args = parser.parse_args(arguments)
    try:
        config = load_retrieval_config(args.retrieval_config)
        if args.top_k is not None:
            config = replace(config, top_k=args.top_k)
        index_path = args.index
        if index_path is None:
            paths = sorted(Path("artifacts/indexes").glob("*/index.json"))
            if len(paths) != 1:
                raise ValueError("Indica --index: se requiere un índice existente inequívoco")
            index_path = paths[0]
        index, artifact = load_index(index_path)
        spec = load_embedding_spec(args.embedding_config)
        if asdict(spec) != index.build_spec["embedding"]:
            raise ValueError("embedding-config es incompatible con el índice")
        embedder = LocalOnnxEmbedder(spec, args.model_directory)
        result = search(args.query, index, artifact.sha256, embedder, config)
    except (ImportError, OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))

    print(f"Query: {result.query}")
    print(f"Query prefix: {result.query_prefix!r}")
    print(f"Model: {spec.model}")
    print(f"Revision: {spec.revision}")
    print(f"Dimensions: {spec.dimensions}")
    print("Algorithm: exhaustive cosine similarity")
    print(f"Vectors evaluated: {len(result.candidates)}")
    print(f"top_k: {config.top_k}")
    print(f"Vector Index ID: {result.vector_index_id}")
    print(f"Artifact SHA-256: {result.artifact_sha256}")
    print(f"Retrieval Config ID: {result.retrieval_config_id}")
    print(f"Retrieval Experiment ID: {result.retrieval_experiment_id}")
    print("\nAll scores (index order, before ranking/Top-K):")
    for candidate in result.candidates:
        print(f"{candidate.chunk_id} score={candidate.score:.9f} chunk={candidate.chunk_index} "
              f"range=[{candidate.start_char},{candidate.end_char})")
    print("\nComplete ranking:")
    for candidate in result.ranking:
        print(f"#{candidate.rank} score={candidate.score:.9f} {candidate.chunk_id}")
    print(f"\nTop-{config.top_k}:")
    for candidate in result.top_results:
        print(f"\n#{candidate.rank} score={candidate.score:.9f}")
        print(f"Document: {candidate.document_id}")
        print(f"Chunk: {candidate.chunk_id} (index {candidate.chunk_index})")
        print(f"Range: [{candidate.start_char},{candidate.end_char})")
        print(f"Preview: {json.dumps(candidate.text[:240], ensure_ascii=True)}")


if __name__ == "__main__":
    main()

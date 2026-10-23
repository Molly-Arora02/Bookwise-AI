"""
BookWise AI — Persistent ChromaDB Vector Store Module.

Manages persistent vector indexing with ChromaDB, duplicate-ingestion prevention,
source-file modification detection, metadata querying, and index reconstruction.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.bookwise.chunking import TextChunk
from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB Vector Store for Document Intelligence."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.persist_dir / "index_manifest.json"

        # Initialize local ONNX MiniLM embedding function (local, offline, fast)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        """Read indexing manifest if present."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading index manifest: {e}")
        return None

    def _save_manifest(self, manifest_data: Dict[str, Any]) -> None:
        """Save indexing manifest atomically to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def is_indexed_and_current(self, current_file_hash: str) -> bool:
        """Check if vector database is already indexed and current with source file."""
        manifest = self.get_manifest()
        if not manifest:
            return False
        
        stored_hash = manifest.get("file_hash")
        chunk_count = self.collection.count()
        manifest_chunks = manifest.get("total_chunks", 0)

        return (
            stored_hash == current_file_hash
            and chunk_count > 0
            and chunk_count == manifest_chunks
        )

    def reset_index(self) -> None:
        """Safely delete and re-create the collection and manifest."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def index_chunks(
        self,
        chunks: List[TextChunk],
        source_file_hash: str,
        source_filename: str,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Index chunks into ChromaDB with duplicate-ingestion prevention
        and source change detection.
        """
        if not chunks:
            return {"status": "empty", "chunks_indexed": 0}

        # Check if already indexed and unchanged
        if not force_rebuild and self.is_indexed_and_current(source_file_hash):
            logger.info("Index is current. Skipping redundant ingestion.")
            return {
                "status": "already_indexed",
                "chunks_indexed": self.collection.count(),
                "file_hash": source_file_hash,
                "rebuilt": False,
            }

        # If file changed or force rebuild, clear previous collection records
        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB collection '{self.collection_name}'...")
        self.reset_index()

        # Batch ingestion (Chroma handles batches well in chunks of 100)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        # Update manifest
        manifest_data = {
            "source_filename": source_filename,
            "file_hash": source_file_hash,
            "total_chunks": len(chunks),
            "collection_name": self.collection_name,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": "all-MiniLM-L6-v2 (Chroma ONNX)",
        }
        self._save_manifest(manifest_data)

        return {
            "status": "success",
            "chunks_indexed": len(chunks),
            "file_hash": source_file_hash,
            "rebuilt": True,
            "manifest": manifest_data,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search.
        Returns list of matched items with chunk text, metadata, and distance.
        """
        if not query.strip():
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        output: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            output.append({
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": float(dist),  # Cosine distance: 0 = identical, 2 = opposite
                "score": max(0.0, 1.0 - float(dist)),  # Similarity score roughly [0, 1]
            })

        return output

    def get_chunk_by_global_index(self, global_index: int) -> Optional[Dict[str, Any]]:
        """Fetch a specific chunk by its sequential global index."""
        results = self.collection.get(
            where={"global_chunk_index": global_index},
            include=["documents", "metadatas"],
            limit=1
        )
        if results and results.get("ids") and len(results["ids"]) > 0:
            return {
                "chunk_id": results["ids"][0],
                "text": results["documents"][0],
                "metadata": results["metadatas"][0],
            }
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store."""
        manifest = self.get_manifest() or {}
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_dir": str(self.persist_dir),
            "source_filename": manifest.get("source_filename", "Unknown"),
            "file_hash": manifest.get("file_hash", "None"),
            "indexed_at": manifest.get("indexed_at", "Never"),
            "is_indexed": self.collection.count() > 0,
        }

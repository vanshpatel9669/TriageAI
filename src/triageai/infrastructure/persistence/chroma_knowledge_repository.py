"""Chroma-backed implementation of VectorSearchRepository.

Chroma persists to a local directory (no server process needed), so this is
real from Phase 2 onward — no Docker required.
"""

from __future__ import annotations

import chromadb

from triageai.domain.models import EvidenceSnippet


class ChromaKnowledgeRepository:
    """Satisfies VectorSearchRepository via a local Chroma collection."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def seed(self, docs: list[dict]) -> None:
        """docs: [{"id": ..., "title": ..., "text": ...}, ...]"""
        if not docs:
            return
        self._collection.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"title": d["title"]} for d in docs],
        )

    async def search_knowledge_docs(self, query: str, top_k: int) -> list[EvidenceSnippet]:
        results = self._collection.query(query_texts=[query], n_results=top_k)

        raw_ids = results["ids"]
        raw_docs = results["documents"]
        raw_metadatas = results["metadatas"]
        raw_distances = results["distances"]

        if not raw_ids or not raw_ids[0]:
            return []
        if raw_docs is None or raw_metadatas is None or raw_distances is None:
            return []

        ids = raw_ids[0]
        docs = raw_docs[0]
        metadatas = raw_metadatas[0]
        distances = raw_distances[0]

        snippets = []
        for doc_id, text, meta, distance in zip(ids, docs, metadatas, distances, strict=True):
            similarity = max(0.0, 1.0 - distance)
            snippets.append(
                EvidenceSnippet(
                    source_id=doc_id,
                    source_type="knowledge_doc",
                    title=str(meta.get("title", doc_id)),
                    text=text,
                    similarity=round(similarity, 4),
                )
            )
        return snippets

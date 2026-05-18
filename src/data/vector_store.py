"""Persistent Chroma knowledge store for KPI metadata retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


class LocalHashEmbeddingFunction(EmbeddingFunction):
    """Small local embedding function so Chroma works without a hosted embedder."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(document) for document in input]

    def _embed(self, document: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_%]+", str(document).lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


class KPIKnowledgeVectorStore:
    """Index and retrieve KPI knowledge-base rows from a persistent Chroma DB."""

    COLLECTION_NAME = "opd_kpi_knowledge"

    def __init__(self, config, data_loader):
        self.config = config
        self.data = data_loader
        self.embedding_function = LocalHashEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(config.vector_store_path))
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "OPD KPI knowledge base rows and KPI catalog"},
        )

    def sync(self, force: bool = False) -> int:
        """Populate Chroma from the loaded knowledge base when needed."""
        documents = list(self._documents())
        if not documents:
            return 0

        existing_count = self.collection.count()
        if existing_count and not force:
            return existing_count

        if existing_count:
            existing = self.collection.get(include=[])
            ids = existing.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)

        self.collection.upsert(
            ids=[item["id"] for item in documents],
            documents=[item["document"] for item in documents],
            metadatas=[item["metadata"] for item in documents],
        )
        return len(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search KPI knowledge with Chroma and return compact result dicts."""
        query = str(query or "").strip()
        if not query:
            return []

        self.sync()
        results = self.collection.query(
            query_texts=[query],
            n_results=max(1, min(int(limit or 5), 10)),
            include=["documents", "metadatas", "distances"],
        )

        found = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for item_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            found.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return found

    def search_kpi(self, kpi_name: str, query: str = "", limit: int = 5) -> list[dict]:
        """Search within records that are explicitly tied to one KPI."""
        kpi_name = str(kpi_name or "").strip()
        if not kpi_name:
            return self.search(query, limit=limit)

        exact_records = self._get_exact_kpi_records(kpi_name, limit=limit)
        if exact_records:
            return exact_records

        query_text = str(query or kpi_name).strip()
        candidates = self.search(query_text, limit=20)
        normalized_kpi = self._normalize(kpi_name)

        exact_matches = []
        document_matches = []
        for result in candidates:
            metadata = result.get("metadata", {})
            result_kpi = self._normalize(metadata.get("kpi", ""))
            document = self._normalize(result.get("document", ""))
            if result_kpi == normalized_kpi:
                exact_matches.append(result)
            elif normalized_kpi in document:
                document_matches.append(result)

        filtered = exact_matches or document_matches
        filtered = filtered[:limit]

        return filtered

    def _get_exact_kpi_records(self, kpi_name: str, limit: int) -> list[dict]:
        self.sync()
        results = self.collection.get(
            where={"kpi": str(kpi_name)},
            include=["documents", "metadatas"],
            limit=max(1, min(int(limit or 5), 10)),
        )

        found = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        for item_id, document, metadata in zip(ids, documents, metadatas):
            found.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": None,
                }
            )
        return found

    def _documents(self) -> Iterable[dict]:
        for sheet_name, sheet in self.data.knowledge_base.items():
            for row_number, (_, row) in enumerate(sheet.iterrows(), start=1):
                values = {
                    str(column): self._clean_value(value)
                    for column, value in row.dropna().to_dict().items()
                }
                values = {key: value for key, value in values.items() if value}
                if not values:
                    continue

                kpi_name = (
                    values.get("KPI_Name")
                    or values.get("KPI")
                    or values.get("Parent_KPI")
                    or values.get("Child_KPI")
                    or ""
                )
                text = "; ".join(f"{key}: {value}" for key, value in values.items())
                yield {
                    "id": f"kb::{sheet_name}::{row_number}",
                    "document": f"Sheet: {sheet_name}; {text}",
                    "metadata": {
                        "source": "knowledge_base",
                        "sheet": sheet_name,
                        "row": row_number,
                        "kpi": str(kpi_name),
                    },
                }

        for index, (kpi_name, item) in enumerate(self.data.kpi_catalog.items(), start=1):
            aliases = ", ".join(item.get("aliases", [])[:12])
            dataset_column = item.get("dataset_column") or ""
            text = (
                f"KPI catalog entry: {kpi_name}; "
                f"dataset column: {dataset_column or 'not directly available'}; "
                f"aliases: {aliases}"
            )
            yield {
                "id": f"catalog::{index}::{self.data.normalize_lookup_text(kpi_name)}",
                "document": text,
                "metadata": {
                    "source": "kpi_catalog",
                    "sheet": "kpi_catalog",
                    "row": 0,
                    "kpi": str(kpi_name),
                    "dataset_column": str(dataset_column),
                },
            }

    @staticmethod
    def _clean_value(value) -> str:
        text = str(value).strip()
        if text.lower() in {"nan", "none", "nat"}:
            return ""
        return text

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()

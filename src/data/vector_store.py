"""Persistent Chroma knowledge store for KPI metadata retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


class LocalHashEmbeddingFunction(EmbeddingFunction):
    """Small local embedding function so Chroma works without a hosted embedder."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.provider_name = "hash"
        self.signature = f"hash:{dimensions}:v1"

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


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding adapter for a local Ollama embedding model."""

    def __init__(self, model: str):
        from langchain_community.embeddings import OllamaEmbeddings

        self.model = model
        self.provider_name = "ollama"
        self.signature = f"ollama:{model}:v1"
        self._embeddings = OllamaEmbeddings(model=model)

    def __call__(self, input: Documents) -> Embeddings:
        return self._embeddings.embed_documents([str(document) for document in input])


class KPIKnowledgeVectorStore:
    """Index and retrieve KPI knowledge-base rows from a persistent Chroma DB."""

    COLLECTION_NAME = "opd_kpi_knowledge"
    COLLECTION_DESCRIPTION = "OPD KPI knowledge base rows and KPI catalog"
    HYBRID_CANDIDATE_LIMIT = 50
    VECTOR_WEIGHT = 0.55
    LEXICAL_WEIGHT = 0.45

    def __init__(self, config, data_loader):
        self.config = config
        self.data = data_loader
        self.embedding_function = self._create_embedding_function()
        self.client = chromadb.PersistentClient(path=str(config.vector_store_path))
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata=self._collection_metadata(),
        )
        self._ensure_embedding_signature()

    def sync(self, force: bool = False) -> int:
        """Populate Chroma from the loaded knowledge base when needed."""
        documents = list(self._documents())
        if not documents:
            return 0

        existing_count = self.collection.count()
        if existing_count and not force:
            return existing_count

        if existing_count:
            self._clear_collection()

        self.collection.upsert(
            ids=[item["id"] for item in documents],
            documents=[item["document"] for item in documents],
            metadatas=[item["metadata"] for item in documents],
        )
        return len(documents)

    def _create_embedding_function(self) -> EmbeddingFunction:
        provider = str(getattr(self.config, "embedding_provider", "auto")).strip().lower()
        model = str(getattr(self.config, "embedding_model", "")).strip()

        if provider in {"hash", "local", "local-hash"}:
            return LocalHashEmbeddingFunction()

        if provider in {"auto", "ollama"}:
            try:
                embedding_function = OllamaEmbeddingFunction(model or "nomic-embed-text")
                embedding_function(["embedding health check"])
                return embedding_function
            except Exception as exc:
                if provider == "ollama":
                    raise RuntimeError(
                        "Ollama embeddings are not available. Start Ollama, pull the "
                        f"'{model or 'nomic-embed-text'}' model, or set "
                        "EMBEDDING_PROVIDER=hash."
                    ) from exc
                print(f"Ollama embeddings not available, using local hash embeddings: {exc}")
                return LocalHashEmbeddingFunction()

        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{provider}'. Use 'auto', 'ollama', or 'hash'."
        )

    def _collection_metadata(self) -> dict:
        return {
            "description": self.COLLECTION_DESCRIPTION,
            "embedding_provider": self.embedding_function.provider_name,
            "embedding_signature": self.embedding_function.signature,
        }

    def _ensure_embedding_signature(self) -> None:
        metadata = dict(self.collection.metadata or {})
        existing_signature = metadata.get("embedding_signature")
        current_signature = self.embedding_function.signature

        if existing_signature == current_signature:
            return

        if self.collection.count():
            self._clear_collection()

        metadata.update(self._collection_metadata())
        self.collection.modify(metadata=metadata)

    def _clear_collection(self) -> None:
        existing = self.collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid vector/lexical search followed by lightweight reranking."""
        query = str(query or "").strip()
        if not query:
            return []

        self.sync()
        result_limit = max(1, min(int(limit or 5), 10))
        candidate_limit = min(
            max(result_limit * 5, 20),
            self.HYBRID_CANDIDATE_LIMIT,
            self.collection.count(),
        )
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=max(1, candidate_limit),
            include=["documents", "metadatas", "distances"],
        )
        lexical_results = self._lexical_candidates(
            query,
            limit=max(candidate_limit, result_limit),
        )

        vector_candidates = []
        ids = vector_results.get("ids", [[]])[0]
        documents = vector_results.get("documents", [[]])[0]
        metadatas = vector_results.get("metadatas", [[]])[0]
        distances = vector_results.get("distances", [[]])[0]
        for item_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            vector_candidates.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return self._hybrid_rerank(
            query,
            vector_candidates,
            lexical_results,
            limit=result_limit,
        )

    def search_kpi(self, kpi_name: str, query: str = "", limit: int = 5) -> list[dict]:
        """Search within records that are explicitly tied to one KPI."""
        kpi_name = str(kpi_name or "").strip()
        if not kpi_name:
            return self.search(query, limit=limit)

        query_text = str(query or kpi_name).strip()
        exact_records = self._get_exact_kpi_records(
            kpi_name,
            limit=self.HYBRID_CANDIDATE_LIMIT,
        )
        if exact_records:
            return self._rerank_records(query_text, exact_records, limit=limit)

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

    def _lexical_candidates(self, query: str, limit: int) -> list[dict]:
        """Rank all stored documents with a compact BM25-style lexical score."""
        stored = self.collection.get(include=["documents", "metadatas"])
        ids = stored.get("ids", [])
        documents = stored.get("documents", [])
        metadatas = stored.get("metadatas", [])
        if not ids:
            return []

        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        tokenized_documents = [self._tokens(document) for document in documents]
        document_count = len(tokenized_documents)
        average_length = (
            sum(len(tokens) for tokens in tokenized_documents) / document_count
            if document_count
            else 1.0
        )
        document_frequency = Counter()
        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))

        scored = []
        for item_id, document, metadata, tokens in zip(
            ids,
            documents,
            metadatas,
            tokenized_documents,
        ):
            score = self._bm25_score(
                query_tokens,
                tokens,
                document_frequency,
                document_count,
                average_length,
            )
            if score <= 0:
                continue
            scored.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": None,
                    "lexical_score": score,
                }
            )

        scored.sort(key=lambda item: item["lexical_score"], reverse=True)
        return scored[: max(1, limit)]

    def _hybrid_rerank(
        self,
        query: str,
        vector_candidates: list[dict],
        lexical_candidates: list[dict],
        limit: int,
    ) -> list[dict]:
        """Merge vector and lexical rankings, then apply exact-match boosts."""
        merged = {}
        vector_count = max(len(vector_candidates), 1)
        lexical_count = max(len(lexical_candidates), 1)

        for rank, item in enumerate(vector_candidates):
            candidate = merged.setdefault(item["id"], dict(item))
            candidate["vector_rank_score"] = 1.0 - (rank / vector_count)

        max_lexical_score = max(
            (item.get("lexical_score", 0.0) for item in lexical_candidates),
            default=1.0,
        )
        for rank, item in enumerate(lexical_candidates):
            candidate = merged.setdefault(item["id"], dict(item))
            candidate.setdefault("distance", item.get("distance"))
            candidate["lexical_rank_score"] = 1.0 - (rank / lexical_count)
            candidate["lexical_score"] = item.get("lexical_score", 0.0)
            candidate["normalized_lexical_score"] = (
                item.get("lexical_score", 0.0) / max_lexical_score
                if max_lexical_score
                else 0.0
            )

        normalized_query = self._normalize(query)
        query_tokens = set(self._tokens(query))
        for candidate in merged.values():
            document = candidate.get("document", "")
            metadata = candidate.get("metadata", {})
            normalized_document = self._normalize(document)
            normalized_kpi = self._normalize(metadata.get("kpi", ""))
            vector_score = candidate.get("vector_rank_score", 0.0)
            lexical_score = candidate.get(
                "normalized_lexical_score",
                candidate.get("lexical_rank_score", 0.0),
            )
            exact_phrase_boost = (
                0.25
                if normalized_query and normalized_query in normalized_document
                else 0.0
            )
            kpi_tokens = set(self._tokens(normalized_kpi))
            kpi_overlap_boost = (
                0.15 * len(query_tokens & kpi_tokens) / len(kpi_tokens)
                if kpi_tokens
                else 0.0
            )
            candidate["hybrid_score"] = (
                self.VECTOR_WEIGHT * vector_score
                + self.LEXICAL_WEIGHT * lexical_score
                + exact_phrase_boost
                + kpi_overlap_boost
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )
        return ranked[: max(1, limit)]

    def _rerank_records(
        self,
        query: str,
        records: list[dict],
        limit: int,
    ) -> list[dict]:
        """Rerank exact-KPI records by lexical relevance to the full question."""
        query_tokens = self._tokens(query)
        if not query_tokens:
            return records[:limit]

        tokenized_documents = [
            self._tokens(record.get("document", "")) for record in records
        ]
        document_frequency = Counter()
        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))
        average_length = (
            sum(len(tokens) for tokens in tokenized_documents) / len(records)
            if records
            else 1.0
        )
        for record, tokens in zip(records, tokenized_documents):
            record["hybrid_score"] = self._bm25_score(
                query_tokens,
                tokens,
                document_frequency,
                len(records),
                average_length,
            )
        return sorted(
            records,
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )[: max(1, limit)]

    @staticmethod
    def _bm25_score(
        query_tokens: list[str],
        document_tokens: list[str],
        document_frequency: Counter,
        document_count: int,
        average_length: float,
    ) -> float:
        if not document_tokens or not document_count:
            return 0.0

        frequencies = Counter(document_tokens)
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (
                    document_count
                    - document_frequency.get(token, 0)
                    + 0.5
                )
                / (document_frequency.get(token, 0) + 0.5)
            )
            length_normalization = frequency + k1 * (
                1.0
                - b
                + b * len(document_tokens) / max(average_length, 1.0)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0) / length_normalization
            )
        return score

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

    @classmethod
    def _tokens(cls, value: str) -> list[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "for",
            "in",
            "is",
            "of",
            "on",
            "the",
            "to",
            "what",
        }
        return [
            token
            for token in cls._normalize(value).split()
            if token and token not in stopwords
        ]

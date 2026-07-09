"""ChromaDB-backed hybrid retriever: dense vectors + BM25, fused with RRF.

- **Dense**: a persistent Chroma collection. Documents are embedded with Chroma's
  built-in MiniLM model (onnxruntime, no torch), so "can they hold me without
  telling me why" matches Article 10 even without keyword overlap.
- **Sparse**: `rank_bm25` over the same documents (falls back to a tiny built-in
  BM25 if the package is missing) — carries exact terms like "warrant",
  "qatl-i-amd", "Article 10-A".
- **Fusion**: Reciprocal Rank Fusion combines the two rankings.

BM25 is rebuilt in memory at load time from the documents Chroma persisted, so
the only thing on disk is the Chroma store.
"""
from __future__ import annotations

import os
from typing import Optional

import re

from .schema import Chunk, Retrieved
from .retriever import _tokenize, _rrf, _make_bm25

COLLECTION = "pakistani_law"

# Explicit citation lookups: "article 6", "art. 10-A", "section 302", "s 420".
# The word before the number decides the statute; the number is normalised to
# the citation_id form ("10-A" / "10 a" -> "10A").
_ARTICLE_REF = re.compile(r"\b(?:articles?|art)\.?\s*(\d+)\s*-?\s*([a-z])?\b",
                          re.IGNORECASE)
_SECTION_REF = re.compile(r"\b(?:sections?|sec|s)\.?\s*(\d+)\s*-?\s*([a-z])?\b",
                          re.IGNORECASE)


def _explicit_ids(query: str) -> list[str]:
    """Citation ids a query names outright, so we can look them up directly
    instead of hoping fuzzy search surfaces them."""
    ids: list[str] = []
    for num, suf in _ARTICLE_REF.findall(query):
        ids.append(f"constitution:art-{num}{suf.upper()}")
    for num, suf in _SECTION_REF.findall(query):
        ids.append(f"ppc:s-{num}{suf.upper()}")
    # de-dup, preserve order
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]
_CHUNK_FIELDS = ("citation_id", "spoken_ref", "text", "source", "unit_type",
                 "unit_number", "clause", "amendment_status", "year",
                 "text_status")


def _embedding_fn():
    from chromadb.utils import embedding_functions
    # Default = all-MiniLM-L6-v2 via onnxruntime; downloaded & cached on first use.
    return embedding_functions.DefaultEmbeddingFunction()


def _doc_for(c: dict) -> str:
    # Index spoken_ref + text so an article number in the query matches even
    # when it isn't in the body.
    return f"{c['spoken_ref']}\n{c['text']}"


def _chunk_from_meta(meta: dict) -> Chunk:
    kw = {k: meta[k] for k in _CHUNK_FIELDS if k in meta and meta[k] is not None}
    return Chunk(**kw)


class ChromaHybridRetriever:
    def __init__(self, collection, persist_dir: str):
        self._col = collection
        self.persist_dir = persist_dir
        # In-memory sparse side, built from the persisted documents.
        got = collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = got["ids"]
        self.docs: list[str] = got["documents"]
        self.metas: list[dict] = got["metadatas"]
        self._pos = {cid: i for i, cid in enumerate(self.ids)}
        self._bm25 = _make_bm25([_tokenize(d) for d in self.docs])

    # ---- build / load ---------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict], persist_dir: str,
              *, batch: int = 256) -> "ChromaHybridRetriever":
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        # Fresh build: drop any prior collection so re-ingest is idempotent.
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION, embedding_function=_embedding_fn(),
                                       metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            col.add(
                ids=[c["citation_id"] for c in part],
                documents=[_doc_for(c) for c in part],
                metadatas=[{k: c[k] for k in _CHUNK_FIELDS
                            if c.get(k) is not None} for c in part],
            )
        return cls(col, persist_dir)

    @classmethod
    def load(cls, persist_dir: str) -> "ChromaHybridRetriever":
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        col = client.get_collection(COLLECTION, embedding_function=_embedding_fn())
        return cls(col, persist_dir)

    def __len__(self) -> int:
        return len(self.ids)

    # ---- query ----------------------------------------------------------
    def search(self, query: str, k: int = 5,
               source: Optional[str] = None,
               min_score: float = 0.0) -> list[Retrieved]:
        """Up to k fused hits. If the best raw BM25 score is below `min_score`,
        return [] so the caller can fall back on an off-topic query.

        A query that names a provision outright ("article 6", "section 302") gets
        that provision pinned to the top by direct id lookup — fuzzy search alone
        buries exact-number lookups because "article" is in every document and a
        bare number is a weak, ambiguous signal."""
        pool = max(k * 4, 20)

        # Direct lookups first: an explicitly-named article/section is on-topic
        # by definition, so it also bypasses the min_score gate.
        pinned = [self._pos[cid] for cid in _explicit_ids(query)
                  if cid in self._pos
                  and not (source and self.metas[self._pos[cid]].get("source") != source)]

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        if not pinned and min_score > 0.0 and (
                not len(bm25_scores) or max(bm25_scores) < min_score):
            return []
        bm25_order = sorted(range(len(self.ids)),
                            key=lambda i: bm25_scores[i], reverse=True)[:pool]

        # Dense side via Chroma (ordered nearest-first).
        res = self._col.query(query_texts=[query], n_results=min(pool, len(self.ids)))
        dense_order = [self._pos[cid] for cid in res["ids"][0] if cid in self._pos]

        fused: dict[int, float] = {}
        for r, i in enumerate(bm25_order):
            fused[i] = fused.get(i, 0.0) + _rrf(r)
        for r, i in enumerate(dense_order):
            fused[i] = fused.get(i, 0.0) + _rrf(r)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        # Pinned ids lead; fused results follow, minus anything already pinned.
        order = pinned + [i for i, _ in ranked if i not in set(pinned)]
        scores = dict(ranked)

        out: list[Retrieved] = []
        for i in order:
            meta = self.metas[i]
            if source and meta.get("source") != source:
                continue
            out.append(Retrieved(chunk=_chunk_from_meta(meta),
                                 score=scores.get(i, 1.0), rank=len(out)))
            if len(out) >= k:
                break
        return out

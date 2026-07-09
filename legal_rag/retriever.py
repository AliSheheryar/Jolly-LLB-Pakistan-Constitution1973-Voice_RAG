"""Hybrid retrieval: sparse (BM25) + dense (embeddings), fused with RRF.

Legal queries hinge on both exact terms ("warrant", "Article 10-A") and
meaning ("can they hold me without telling me why"), so we run both and fuse.
Dense retrieval is optional and degrades gracefully to BM25-only if
sentence-transformers isn't installed — the pipeline still works.
"""
from __future__ import annotations

import re
from typing import Optional

from .schema import Chunk, Retrieved


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _MiniBM25:
    """Tiny pure-Python BM25 used only if `rank_bm25` isn't installed, so the
    package runs with zero external deps. Install rank-bm25 for the tuned one."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        import math
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.N = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in corpus:
            seen: dict[str, int] = {}
            for t in doc:
                seen[t] = seen.get(t, 0) + 1
            self.tf.append(seen)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for i in range(self.N):
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            scores[i] = s
        return scores


def _make_bm25(corpus_tokens: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(corpus_tokens)
    except Exception:
        return _MiniBM25(corpus_tokens)


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion weight for a given 0-based rank."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], *, embed: bool = True,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.embed = embed
        self.model_name = model_name
        self._bm25 = None
        self._model = None
        self._embeddings = None  # numpy array [N, D] or None
        self._corpus_tokens: list[list[str]] = []

    # ---- indexing -------------------------------------------------------
    def fit(self) -> "HybridRetriever":
        # Index over statute text + its spoken reference so an article number
        # in the query ("10-A") matches even if it's not in the body text.
        docs = [f"{c.spoken_ref}\n{c.text}" for c in self.chunks]
        self._corpus_tokens = [_tokenize(d) for d in docs]
        self._bm25 = _make_bm25(self._corpus_tokens)

        if self.embed:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._embeddings = self._model.encode(
                    docs, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as exc:  # missing lib, no model cache, etc.
                print(f"[retriever] dense retrieval disabled ({exc}); "
                      f"falling back to BM25 only.")
                self.embed = False
        return self

    # pickling: drop the (large, non-picklable) live model; re-load lazily
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def _ensure_model(self):
        if self.embed and self._model is None and self._embeddings is not None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    # ---- query ----------------------------------------------------------
    def search(self, query: str, k: int = 5,
               source: Optional[str] = None,
               min_score: float = 0.0) -> list[Retrieved]:
        """Return up to k fused hits. If the best raw BM25 score is below
        `min_score`, return [] so the pipeline falls back rather than answering
        an off-topic question. Default 0.0 disables the gate."""
        if self._bm25 is None:
            raise RuntimeError("retriever not fitted; call fit() or load an index")

        pool = k * 4  # over-fetch per method before fusion

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        if min_score > 0.0 and (not len(bm25_scores) or max(bm25_scores) < min_score):
            return []
        bm25_order = sorted(range(len(self.chunks)),
                            key=lambda i: bm25_scores[i], reverse=True)[:pool]

        dense_order: list[int] = []
        if self.embed and self._embeddings is not None:
            self._ensure_model()
            import numpy as np
            q = self._model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True)[0]
            sims = self._embeddings @ q
            dense_order = list(np.argsort(-sims)[:pool])

        # Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        for r, i in enumerate(bm25_order):
            fused[i] = fused.get(i, 0.0) + _rrf(r)
        for r, i in enumerate(dense_order):
            fused[i] = fused.get(int(i), 0.0) + _rrf(r)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        out: list[Retrieved] = []
        for rank, (i, score) in enumerate(ranked):
            c = self.chunks[i]
            if source and c.source != source:
                continue
            out.append(Retrieved(chunk=c, score=score, rank=rank))
            if len(out) >= k:
                break
        return out

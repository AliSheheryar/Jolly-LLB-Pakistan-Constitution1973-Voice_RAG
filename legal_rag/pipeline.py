"""End-to-end: query -> retrieve -> generate -> verify -> answer.

    pipeline = LegalRAG.from_index("index", llm)
    answer = pipeline.answer("can you search my car without a warrant?")
    if answer.ok:
        speak(answer.spoken_text)

The `answer.spoken_text` is safe to send to TTS: it is either a verified,
source-grounded reply or a safe fallback. Nothing unverified reaches the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Answer
from .retriever import HybridRetriever
from .generator import generate
from .verifier import verify, render_for_tts, VerificationError
from .llm import LLM


SAFE_FALLBACK = ("I'm not able to give you a reliable answer on that from the "
                 "law I have on hand. Please consult a qualified lawyer.")

RETRY_DIRECTIVE = ("Your previous answer cited a source that was not in the "
                   "provided list, or stated law without a marker. Answer again "
                   "using ONLY the given sources and a [[CIT:<id>]] marker on "
                   "every legal statement, or say you have no provision.")


@dataclass
class LegalRAG:
    retriever: HybridRetriever
    llm: LLM
    k: int = 5
    source_filter: str | None = None  # e.g. restrict to the Constitution
    min_score: float = 0.0            # BM25 relevance floor; >0 enables gating

    @classmethod
    def from_index(cls, index_dir: str, llm: LLM, **kw) -> "LegalRAG":
        from .vectorstore import ChromaHybridRetriever
        return cls(retriever=ChromaHybridRetriever.load(index_dir), llm=llm, **kw)

    def answer(self, query: str) -> Answer:
        hits = self.retriever.search(query, k=self.k, source=self.source_filter,
                                     min_score=self.min_score)
        if not hits:
            return Answer(ok=False, spoken_text=SAFE_FALLBACK,
                          reason="no retrieval hits")

        directive = ""
        last_reason = ""
        for attempt in range(2):  # one generation + one stricter retry
            marked = generate(self.llm, query, hits, extra_directive=directive)
            try:
                cites = verify(marked, hits)
            except VerificationError as e:
                last_reason = e.reason
                directive = RETRY_DIRECTIVE
                continue
            return Answer(
                ok=True,
                spoken_text=render_for_tts(marked, hits),
                marked_text=marked,
                citations=cites,
            )

        return Answer(ok=False, spoken_text=SAFE_FALLBACK,
                      reason=f"verification failed: {last_reason}")


def _demo() -> None:
    """Offline smoke test using the sample corpus and the stub LLM."""
    import os
    from .ingest import build
    from .llm import StubLLM

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(here, "corpus", "pakistan_constitution.sample.jsonl")
    retriever = build([corpus], os.path.join(here, "index"),
                      allow_non_verbatim=True, embed=False)

    rag = LegalRAG(retriever=retriever, llm=StubLLM(), k=3, min_score=2.0)
    for q in [
        "can you arrest me without telling me the grounds?",
        "am I entitled to a lawyer?",
        "what's the speed limit on the motorway?",  # not in corpus -> fallback
    ]:
        a = rag.answer(q)
        print(f"\nQ: {q}\n  ok={a.ok} cites={a.citations}\n  > {a.spoken_text}")


if __name__ == "__main__":
    _demo()

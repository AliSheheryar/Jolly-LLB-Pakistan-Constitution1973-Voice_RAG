"""Core data models for the legal RAG pipeline.

A `Chunk` is one atomic, citable unit of law — a single article, section, or
clause. Chunks are NEVER split across a clause boundary. The `text` field must
hold the *verbatim* statutory text in production; the pipeline quotes it, it
does not paraphrase from the model's memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    # Stable, machine-checkable citation id, e.g. "constitution:art-10A".
    # This is the token the LLM must echo and the verifier checks against.
    citation_id: str

    # How the citation is spoken aloud, e.g. "Article 10-A of the Constitution".
    spoken_ref: str

    # Verbatim statutory text. MUST be the official text in production.
    text: str

    # Provenance / filtering metadata.
    source: str                       # "Constitution of Pakistan 1973"
    unit_type: str                    # "article" | "section" | "clause"
    unit_number: str                  # "10A", "10(1)", "54"
    clause: Optional[str] = None      # sub-clause if this chunk is a clause
    amendment_status: str = "in force"
    year: Optional[int] = None

    # Guards against fabricated statute text sneaking into the corpus.
    # "verbatim" = official text loaded; anything else is treated as unusable
    # for citation and is filtered out of retrieval by default.
    text_status: str = "verbatim"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


@dataclass
class Retrieved:
    """A chunk plus the score that surfaced it."""
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class Answer:
    """Final result handed to the voice layer."""
    ok: bool                          # False => fell back to a safe non-answer
    spoken_text: str                  # ready for TTS (citation markers rendered)
    marked_text: str = ""             # raw LLM text with [[CIT:...]] markers
    citations: list[str] = field(default_factory=list)  # citation_ids used
    reason: str = ""                  # why it failed, if ok is False

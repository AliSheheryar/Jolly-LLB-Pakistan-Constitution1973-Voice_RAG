"""Grounded generation.

Builds the tightly-constrained prompt and calls the LLM. The model may only
speak law that appears in the retrieved context, and must tag every legal
proposition with the source's [[CIT:<citation_id>]] marker so the verifier can
check it. If nothing fits, it must say so rather than invent a provision.
"""
from __future__ import annotations

from .schema import Retrieved
from .llm import LLM


SYSTEM_PROMPT = """\
You are a voice assistant that answers questions about Pakistani law for a \
caller who may be a police official. You provide legal INFORMATION, not legal \
advice, and you speak in short, clear sentences suitable for being read aloud \
on a phone call.

ABSOLUTE RULES:
1. Use ONLY the provisions in the <sources> block below. Do not rely on any \
other legal knowledge. If the sources do not cover the question, say exactly: \
"I don't have a specific provision on that." Do not guess.
2. Every time you state what the law says, immediately append the marker \
[[CIT:<id>]] using the exact id of the source you are relying on. Never write \
an article or section number that is not backed by such a marker.
3. Quote or closely paraphrase the source text; do not embellish it.
4. Be brief — this is spoken aloud. Two or three sentences is usually enough.
5. You may remind the caller this is general legal information and that a \
lawyer or court decides how it applies. Do not argue, escalate, or give \
tactical instructions.
"""


def _format_sources(hits: list[Retrieved]) -> str:
    lines = []
    for h in hits:
        c = h.chunk
        lines.append(
            f'<source id="{c.citation_id}" ref="{c.spoken_ref}" '
            f'status="{c.amendment_status}">\n{c.text}\n</source>'
        )
    return "<sources>\n" + "\n".join(lines) + "\n</sources>"


def build_user_prompt(query: str, hits: list[Retrieved]) -> str:
    # The [[ID:...]] hints let the offline StubLLM find valid ids; a real model
    # reads the id="" attributes. Both resolve to the same citation_ids.
    id_hints = " ".join(f"[[ID:{h.chunk.citation_id}]]" for h in hits)
    return (
        f"{_format_sources(hits)}\n\n"
        f"Caller said: \"{query}\"\n\n"
        f"Answer using only the sources above, tagging each legal statement "
        f"with its [[CIT:<id>]] marker. Valid ids: {id_hints}"
    )


def generate(llm: LLM, query: str, hits: list[Retrieved],
             *, extra_directive: str = "") -> str:
    system = SYSTEM_PROMPT + (f"\n\n{extra_directive}" if extra_directive else "")
    return llm.complete(system, build_user_prompt(query, hits)).strip()

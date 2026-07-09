"""Citation verification — the safety gate before anything is spoken.

Two checks:
  1. Every [[CIT:<id>]] marker must reference a chunk that was actually
     retrieved for this query. A marker to an unknown id = hallucinated source.
  2. Soft check: any article/section number spoken in the prose must be backed
     by a marker. A bare "Article 21" with no marker is suspicious and fails.

If verification fails the pipeline retries once, then falls back to a safe
non-answer. Only verified text is ever rendered for TTS.
"""
from __future__ import annotations

import re

from .schema import Retrieved


_MARKER = re.compile(r"\[\[CIT:(?P<id>[^\]]+)\]\]")
# Matches "Article 10A", "article 10-A", "section 54", "Art. 13(1)" and captures
# the number ("10", "10A") so we can check it against what was actually cited.
_LEGAL_REF = re.compile(
    r"\b(?:article|art\.?|section|sec\.?)\s*(\d+)\s*-?\s*([a-z])?(?:\([\da-z]+\))?",
    re.IGNORECASE)


def _ref_key(num: str, suffix: str = "") -> str:
    """Canonical key for a provision number: '10','A' -> '10a'; '302' -> '302'."""
    return f"{num}{suffix}".lower()


def _cited_numbers(cited: list[str], hits: list[Retrieved]) -> set[str]:
    """The set of provision numbers the answer actually cited, in both full
    ('10a') and digits-only ('10') form, so a prose mention of the same
    provision counts as backed."""
    by_id = {h.chunk.citation_id: h.chunk for h in hits}
    keys: set[str] = set()
    for cid in cited:
        c = by_id.get(cid)
        if not c:
            continue
        num = c.unit_number.lower()
        keys.add(num)                                   # "10a", "302"
        m = re.match(r"\d+", num)
        if m:
            keys.add(m.group(0))                        # digits only: "10"
    return keys


class VerificationError(Exception):
    def __init__(self, reason: str, bad_ids: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.bad_ids = bad_ids or []


def extract_citations(marked_text: str) -> list[str]:
    return _MARKER.findall(marked_text)


def verify(marked_text: str, hits: list[Retrieved]) -> list[str]:
    """Return the list of valid citation_ids used, or raise VerificationError."""
    allowed = {h.chunk.citation_id for h in hits}

    cited = extract_citations(marked_text)
    bad = [c for c in cited if c not in allowed]
    if bad:
        raise VerificationError(
            f"cited source(s) not in retrieved set: {bad}", bad_ids=bad)

    # An explicit "no provision" answer is allowed to carry no citations.
    if not cited:
        if "don't have a specific provision" in marked_text.lower():
            return []
        raise VerificationError("answer states law but cites no source")

    # Soft check: any article/section number named in the prose must correspond
    # to a provision the answer actually cited. This catches a hallucinated
    # "Article 21" (a number that was never retrieved/cited) while allowing the
    # model to restate the number of a provision it did cite ("Article 6 …
    # [[CIT:constitution:art-6]]").
    stripped = _MARKER.sub(" ", marked_text)
    backed = _cited_numbers(cited, hits)
    for num, suffix in _LEGAL_REF.findall(stripped):
        key = _ref_key(num, suffix)
        if key not in backed and num not in backed:
            raise VerificationError(
                f"unbacked legal reference in prose: {num}{suffix}".strip())

    return cited


def render_for_tts(marked_text: str, hits: list[Retrieved]) -> str:
    """Replace [[CIT:id]] markers with their spoken reference for playback."""
    ref = {h.chunk.citation_id: h.chunk.spoken_ref for h in hits}

    def repl(m: re.Match) -> str:
        cid = m.group("id")
        return f"(under {ref.get(cid, 'the cited provision')})"

    text = _MARKER.sub(repl, marked_text)
    return re.sub(r"\s{2,}", " ", text).strip()

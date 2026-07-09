"""Download and parse Pakistani statutes into a verbatim JSONL corpus.

Source: pakistani.org, which publishes the Constitution of Pakistan 1973 and the
Pakistan Penal Code 1860 as structured HTML. Both use the same markup for a
citable unit:

    <tr>
      <td><nobr><b>10A</b></nobr></td>          <- unit number (bold, no parens)
      <td><b>Right to fair trial.</b><br> ...body... </td>
    </tr>

Clause markers inside the body ("(1)", "(a)") are also <nobr> but are NOT bold
and carry parentheses, so they never get mistaken for a unit number. Footnote
superscripts (amendment reference digits) are stripped; the bracketed amendment
*text* they point at ("[Majlis-e-Shoora (Parliament)]") is kept — that is the
current, in-force wording.

The result is one JSON object per citable unit, matching `schema.Chunk`, with
`text_status="verbatim"` so it is usable for citation in production.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass

BASE = "http://www.pakistani.org"
UA = "Mozilla/5.0 (compatible; legal-rag/1.0; +https://github.com/)"
RAW_DIR_DEFAULT = "data/raw"

# unit number: "9", "10A", "175B", "462G", and PPC's dotted form "302."
_UNIT_NUM = re.compile(r"^\d+[A-Z]?\.?$")
_WS = re.compile(r"\s+")
_EMPTY_BRACKET = re.compile(r"\[\s*\]")         # left over where an amendment omitted text


@dataclass(frozen=True)
class SourceSpec:
    key: str                 # "constitution" | "ppc"
    source: str              # human-readable source name stored on each chunk
    year: int
    unit_type: str           # "article" | "section"
    cid_prefix: str          # "constitution:art-" | "ppc:s-"
    spoken_unit: str         # "Article" | "Section"
    spoken_source: str       # "of the Constitution" | "of the Pakistan Penal Code"


CONSTITUTION = SourceSpec(
    key="constitution",
    source="Constitution of Pakistan 1973",
    year=1973,
    unit_type="article",
    cid_prefix="constitution:art-",
    spoken_unit="Article",
    spoken_source="of the Constitution",
)

PPC = SourceSpec(
    key="ppc",
    source="Pakistan Penal Code 1860",
    year=1860,
    unit_type="section",
    cid_prefix="ppc:s-",
    spoken_unit="Section",
    spoken_source="of the Pakistan Penal Code",
)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _cached(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, encoding="utf-8", errors="replace").read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    html = _get(url)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(html)
    time.sleep(0.4)  # be polite to the origin
    return html


def _constitution_part_urls(index_html: str) -> list[str]:
    """Pull every Constitution part/preamble page link from the index page."""
    hrefs = re.findall(r'href="(/pakistan/constitution/[^"]+\.html)"', index_html)
    seen, out = set(), []
    for h in hrefs:
        if "amendments/" in h or h.endswith("copyright.html"):
            continue
        if not re.search(r"/(preamble|part\d)", h):
            continue
        if h not in seen:
            seen.add(h)
            out.append(BASE + h)
    return out


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #
def _spoken_num(num: str) -> str:
    # "10A" -> "10-A" so TTS reads "ten-A" not "tenA"; "375" stays "375".
    return re.sub(r"^(\d+)([A-Z])$", r"\1-\2", num)


def _clean(text: str) -> str:
    text = _EMPTY_BRACKET.sub("", text)
    return _WS.sub(" ", text).strip()


def _parse_page(html: str, spec: SourceSpec) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Footnote superscripts are amendment *reference numbers*, not law text.
    for sup in soup.find_all("sup"):
        sup.decompose()

    units: list[dict] = []
    for nobr in soup.find_all("nobr"):
        b = nobr.find("b", recursive=False) or nobr.find("b")
        if not b:
            continue
        num = b.get_text(strip=True)
        if not _UNIT_NUM.match(num):
            continue
        num = num.rstrip(".")           # PPC prints "302."; normalise to "302"

        num_td = nobr.find_parent("td")
        if num_td is None:
            continue
        body_td = num_td.find_next_sibling("td")
        if body_td is None:
            continue

        # Marginal note / title is the first <b> in the body cell.
        title_b = body_td.find("b")
        title = title_b.get_text(" ", strip=True) if title_b else ""
        title = title.rstrip(" .:")
        if title_b:
            title_b.decompose()  # drop so it isn't duplicated in the body text

        body = _clean(body_td.get_text(" ", strip=True))
        if len(body) < 3:
            continue  # omitted / repealed placeholder ("* * *")

        text = f"{title}. {body}" if title else body
        spoken = f"{spec.spoken_unit} {_spoken_num(num)} {spec.spoken_source}"
        units.append({
            "citation_id": f"{spec.cid_prefix}{num}",
            "spoken_ref": spoken,
            "text": text,
            "source": spec.source,
            "unit_type": spec.unit_type,
            "unit_number": num,
            "amendment_status": "in force",
            "year": spec.year,
            "text_status": "verbatim",
        })
    return units


def _dedupe(units: list[dict]) -> list[dict]:
    """Keep the first occurrence of each citation_id (nav/footers can repeat)."""
    seen, out = set(), []
    for u in units:
        if u["citation_id"] in seen:
            continue
        seen.add(u["citation_id"])
        out.append(u)
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def fetch_constitution(raw_dir: str = RAW_DIR_DEFAULT) -> list[dict]:
    index = _cached(BASE + "/pakistan/constitution/",
                    os.path.join(raw_dir, "const_index.html"))
    units: list[dict] = []
    for url in _constitution_part_urls(index):
        name = url.rsplit("/", 1)[-1]
        html = _cached(url, os.path.join(raw_dir, "constitution", name))
        units.extend(_parse_page(html, CONSTITUTION))
    return _dedupe(units)


def fetch_ppc(raw_dir: str = RAW_DIR_DEFAULT) -> list[dict]:
    html = _cached(BASE + "/pakistan/legislation/1860/actXLVof1860.html",
                   os.path.join(raw_dir, "ppc.html"))
    return _dedupe(_parse_page(html, PPC))


FETCHERS = {"constitution": fetch_constitution, "ppc": fetch_ppc}
OUT_FILES = {
    "constitution": "constitution_1973.jsonl",
    "ppc": "ppc_1860.jsonl",
}


def download(which: list[str], corpus_dir: str = "corpus",
             raw_dir: str = RAW_DIR_DEFAULT) -> dict[str, int]:
    os.makedirs(corpus_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for key in which:
        units = FETCHERS[key](raw_dir)
        out = os.path.join(corpus_dir, OUT_FILES[key])
        with open(out, "w", encoding="utf-8") as fh:
            for u in units:
                fh.write(json.dumps(u, ensure_ascii=False) + "\n")
        counts[key] = len(units)
        print(f"[fetch] {key}: {len(units)} units -> {out}")
    return counts

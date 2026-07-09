"""Ingestion: JSONL corpus -> validated chunks -> persisted retriever index.

Run once whenever the corpus changes:

    python -m legal_rag.ingest corpus/*.jsonl --out index/

The corpus format is one JSON object per line, each matching `Chunk`. Chunking
is done *at authoring time* (one article/section/clause per line) so the legal
unit stays intact — we deliberately do not auto-split running text here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from typing import Iterable

from .schema import Chunk
from .retriever import HybridRetriever


REQUIRED = {"citation_id", "spoken_ref", "text", "source", "unit_type", "unit_number"}


def load_jsonl(paths: Iterable[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                obj = json.loads(line)
                missing = REQUIRED - obj.keys()
                if missing:
                    raise ValueError(f"{path}:{lineno} missing fields: {missing}")
                if obj["citation_id"] in seen:
                    raise ValueError(f"{path}:{lineno} duplicate citation_id "
                                     f"{obj['citation_id']!r}")
                seen.add(obj["citation_id"])
                chunks.append(Chunk.from_dict(obj))
    return chunks


def build(paths: Iterable[str], out_dir: str, *, allow_non_verbatim: bool = False,
          embed: bool = True) -> HybridRetriever:
    chunks = load_jsonl(paths)

    if not allow_non_verbatim:
        usable = [c for c in chunks if c.text_status == "verbatim"]
        dropped = len(chunks) - len(usable)
        if dropped:
            print(f"[ingest] WARNING: dropped {dropped} chunk(s) whose text_status "
                  f"is not 'verbatim'. Load official text before production use, "
                  f"or pass allow_non_verbatim=True for dev.")
        chunks = usable

    if not chunks:
        raise SystemExit("[ingest] no usable chunks — nothing to index.")

    retriever = HybridRetriever(chunks, embed=embed)
    retriever.fit()

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retriever.pkl"), "wb") as fh:
        pickle.dump(retriever, fh)
    print(f"[ingest] indexed {len(chunks)} chunks -> {out_dir}/retriever.pkl")
    return retriever


def load_index(out_dir: str) -> HybridRetriever:
    with open(os.path.join(out_dir, "retriever.pkl"), "rb") as fh:
        return pickle.load(fh)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="JSONL corpus files or globs")
    ap.add_argument("--out", default="index", help="output index directory")
    ap.add_argument("--allow-non-verbatim", action="store_true",
                    help="dev only: index paraphrase/placeholder chunks too")
    ap.add_argument("--no-embed", action="store_true",
                    help="BM25 only (skip dense embeddings)")
    args = ap.parse_args()

    paths: list[str] = []
    for g in args.globs:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        raise SystemExit("no corpus files matched")

    build(paths, args.out,
          allow_non_verbatim=args.allow_non_verbatim,
          embed=not args.no_embed)


if __name__ == "__main__":
    _main()

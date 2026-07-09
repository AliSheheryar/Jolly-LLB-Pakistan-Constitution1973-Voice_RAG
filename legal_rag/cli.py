"""Command-line interface for the Pakistani-law RAG.

    python -m legal_rag.cli download                 # fetch Constitution + PPC
    python -m legal_rag.cli ingest                   # build the Chroma index
    python -m legal_rag.cli query "can you search my car without a warrant?"
    python -m legal_rag.cli chat                      # interactive REPL

Generation uses OpenRouter's Llama 3.1 8B (set OPENROUTER_API_KEY). Pass --stub
to run the whole pipeline offline with a deterministic stand-in LLM (no key,
no network) — handy for smoke-testing retrieval + verification.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from .ingest import load_jsonl
from .vectorstore import ChromaHybridRetriever
from .pipeline import LegalRAG
from .llm import OpenRouterLLM, StubLLM

DEFAULT_CORPUS = "corpus"
DEFAULT_INDEX = "index"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


# --------------------------------------------------------------------------- #
def _corpus_files(corpus_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl")))
    if not files:
        sys.exit(f"no .jsonl corpus files in {corpus_dir!r}; run `download` first")
    return files


def cmd_download(args: argparse.Namespace) -> None:
    from .fetch import download
    which = args.only.split(",") if args.only else ["constitution", "ppc"]
    counts = download(which, corpus_dir=args.corpus, raw_dir=args.raw)
    print(f"[download] done: {sum(counts.values())} units across {len(counts)} source(s)")


def cmd_ingest(args: argparse.Namespace) -> None:
    chunks = load_jsonl(_corpus_files(args.corpus))
    if not args.allow_non_verbatim:
        usable = [c for c in chunks if c.text_status == "verbatim"]
        dropped = len(chunks) - len(usable)
        if dropped:
            print(f"[ingest] dropped {dropped} non-verbatim chunk(s); "
                  f"pass --allow-non-verbatim to keep them (dev only).")
        chunks = usable
    if not chunks:
        sys.exit("[ingest] no usable chunks — nothing to index.")
    print(f"[ingest] embedding {len(chunks)} chunks into Chroma at {args.index!r} "
          f"(first run downloads the MiniLM model)…")
    r = ChromaHybridRetriever.build([c.to_dict() for c in chunks], args.index)
    print(f"[ingest] indexed {len(r)} chunks -> {args.index}/ (dense + BM25 hybrid)")


def _build_rag(args: argparse.Namespace) -> LegalRAG:
    if not os.path.isdir(args.index) or not os.listdir(args.index):
        sys.exit(f"index {args.index!r} is empty; run `ingest` first")
    llm = StubLLM() if args.stub else OpenRouterLLM(model=args.model)
    return LegalRAG.from_index(
        args.index, llm, k=args.k,
        source_filter=args.source or None,
        min_score=args.min_score,
    )


def _print_answer(query: str, ans) -> None:
    print(f"\nQ: {query}")
    print(f"A: {ans.spoken_text}")
    if ans.ok:
        print(f"   [cited: {', '.join(ans.citations) or 'none'}]")
    else:
        print(f"   [fallback: {ans.reason}]")


def cmd_query(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    _print_answer(args.text, rag.answer(args.text))


def cmd_chat(args: argparse.Namespace) -> None:
    rag = _build_rag(args)
    print("Pakistani-law assistant. Ask a question; Ctrl-D or 'exit' to quit.\n"
          "This is legal information, not advice.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        _print_answer(q, rag.answer(q))


# --------------------------------------------------------------------------- #
def _add_query_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", default=DEFAULT_INDEX, help="Chroma index directory")
    p.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    p.add_argument("--source", default="", help="restrict to one source, e.g. "
                   "'Pakistan Penal Code 1860'")
    p.add_argument("--min-score", dest="min_score", type=float, default=0.0,
                   help="BM25 relevance floor; >0 drops off-topic queries")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    p.add_argument("--stub", action="store_true",
                   help="offline deterministic LLM (no API key / network)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="legal_rag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch statutes into corpus/*.jsonl")
    d.add_argument("--only", default="", help="comma list: constitution,ppc")
    d.add_argument("--corpus", default=DEFAULT_CORPUS)
    d.add_argument("--raw", default="data/raw")
    d.set_defaults(func=cmd_download)

    g = sub.add_parser("ingest", help="build the Chroma hybrid index")
    g.add_argument("--corpus", default=DEFAULT_CORPUS)
    g.add_argument("--index", default=DEFAULT_INDEX)
    g.add_argument("--allow-non-verbatim", action="store_true")
    g.set_defaults(func=cmd_ingest)

    q = sub.add_parser("query", help="answer one question and exit")
    q.add_argument("text")
    _add_query_opts(q)
    q.set_defaults(func=cmd_query)

    c = sub.add_parser("chat", help="interactive question loop")
    _add_query_opts(c)
    c.set_defaults(func=cmd_chat)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

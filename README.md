![Jolly LLB — Pakistani-law RAG](assets/jolly-llb-frontend.png)

# Legal RAG (Pakistani law) — CLI

A retrieval-augmented question-answering CLI over **real Pakistani statutes**.
It retrieves the relevant law with a **hybrid vector + BM25 search**, generates a
grounded answer with **Llama 3.1 (via OpenRouter)**, and **verifies every
citation** against the retrieved text before showing it. Nothing the model
asserts about the law reaches you unless it maps to a provision that was actually
retrieved.

```
question
    │
    ▼
ChromaHybridRetriever      dense vectors (Chroma/MiniLM) + BM25, fused with RRF
    │  top-k provisions
    ▼
generator (Llama 3.1)      grounded prompt: cite ONLY retrieved law,
    │  marked text            tag every legal statement with [[CIT:<id>]]
    ▼
verifier                   every marker must map to a retrieved chunk;
    │  ok / fail             bare article/section numbers with no marker fail
    ▼
answer  (verified, cited)  or a safe "I don't have a provision on that" fallback
```

On verification failure the pipeline retries once with a stricter directive, then
falls back to a safe non-answer.

## Corpus

Downloaded and parsed from [pakistani.org](http://www.pakistani.org) into verbatim,
citable units (`text_status: "verbatim"`), one JSON object per line:

| Source | Units | File |
|--------|-------|------|
| Constitution of Pakistan 1973 | 323 articles | `corpus/constitution_1973.jsonl` |
| Pakistan Penal Code 1860 | 528 sections | `corpus/ppc_1860.jsonl` |

`legal_rag/fetch.py` does the download + HTML parsing (article/section number,
marginal-note title, clause structure; amendment footnote markers stripped,
in-force bracketed text kept). Add more statutes by dropping more `*.jsonl` in
`corpus/` and re-running `ingest`.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...        # https://openrouter.ai/keys
```

## Use

```bash
# 1. Download the statutes into corpus/*.jsonl  (cached under data/raw/)
python -m legal_rag.cli download

# 2. Build the hybrid index (Chroma vector store + BM25). First run downloads
#    the MiniLM embedding model (~80 MB, onnxruntime — no torch).
python -m legal_rag.cli ingest

# 3. Ask a question (Llama 3.1 8B via OpenRouter)
python -m legal_rag.cli query "can the police search my car without a warrant?"

# 4. Or an interactive session
python -m legal_rag.cli chat
```

Useful flags on `query` / `chat`:

| Flag | Meaning |
|------|---------|
| `-k N` | how many provisions to retrieve (default 5) |
| `--source "Pakistan Penal Code 1860"` | restrict to one statute |
| `--min-score F` | BM25 relevance floor; drops off-topic questions before the LLM |
| `--model <id>` | any OpenRouter model id (default `meta-llama/llama-3.1-8b-instruct`) |
| `--stub` | run fully offline with a deterministic stand-in LLM (no key/network) — smoke-tests retrieval + verification |

## Layout

| File | Role |
|------|------|
| `fetch.py` | download + parse statutes → verbatim JSONL corpus |
| `schema.py` | `Chunk` (one citable unit of law), `Retrieved`, `Answer` |
| `ingest.py` | JSONL corpus → validated chunks |
| `vectorstore.py` | `ChromaHybridRetriever` — dense (Chroma) + BM25, RRF fusion |
| `generator.py` | strict grounded prompt + context assembly |
| `verifier.py` | citation validation + render markers for display |
| `llm.py` | `OpenRouterLLM` (Llama 3.1) and `StubLLM` (offline) |
| `pipeline.py` | `LegalRAG.answer()` orchestration |
| `cli.py` | `download` / `ingest` / `query` / `chat` |

## Design choices worth knowing

- **Hybrid retrieval.** Legal queries hinge on both exact terms ("warrant",
  "qatl-i-amd", "Article 10-A") and meaning ("can they hold me without telling me
  why"), so we run BM25 and dense vectors and fuse with Reciprocal Rank Fusion.
- **Verify-then-answer.** The LLM tags each legal claim with `[[CIT:<id>]]`; the
  verifier checks every id against the retrieved set and rejects bare
  article/section numbers that carry no marker — a hallucinated citation can't
  reach the user.
- **Verbatim only.** `ingest` drops any chunk whose `text_status` isn't
  `verbatim`, so paraphrase/placeholder text can never be cited.
- **This is legal information, not advice.** The prompt enforces that framing.

## Not included yet

- **Voice.** The Twilio ConversationRelay server (`legal_rag/server.py`) is left
  in place but is not wired to this CLI — voice is a later step.
- **CrPC 1898 / Evidence Order** — no clean machine-readable source on
  pakistani.org; add them as `*.jsonl` when a good source is available.

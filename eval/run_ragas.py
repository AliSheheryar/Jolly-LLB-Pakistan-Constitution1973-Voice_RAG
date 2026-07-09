"""Evaluate the Legal-RAG pipeline against eval/golden_dataset.jsonl with RAGAS.

Two layers of scoring:

  1. Deterministic retrieval metrics (NO LLM needed) — hit@k and MRR of the
     golden `expected_citations` in the retrieved set. Always runs.

  2. RAGAS metrics (needs a judge LLM) — faithfulness, answer_relevancy,
     context_precision, context_recall. Skipped with a clear message if no
     OPENROUTER_API_KEY is set.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    .venv/bin/python -m eval.run_ragas                 # generate answers live
    .venv/bin/python -m eval.run_ragas --judge openai/gpt-4o-mini --k 5

The generator LLM (produces the RAG answers) and the judge LLM (scores them)
both route through OpenRouter. Embeddings use Chroma's local ONNX MiniLM so no
torch / sentence-transformers is required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = HERE / "golden_dataset.jsonl"


# ---------------------------------------------------------------------------
# 1. Run the pipeline over the golden questions -> per-row eval records
# ---------------------------------------------------------------------------
def build_records(k: int, gen_model: str) -> list[dict]:
    from legal_rag.pipeline import LegalRAG
    from legal_rag.llm import OpenRouterLLM

    llm = OpenRouterLLM(model=gen_model)
    rag = LegalRAG.from_index(str(ROOT / "index"), llm, k=k)

    rows = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    records = []
    for r in rows:
        q = r["question"]
        # retrieval layer (independent of generation, so we score it directly)
        hits = rag.retriever.search(q, k=k)
        contexts = [h.chunk.text for h in hits]
        retrieved_ids = [h.chunk.citation_id for h in hits]
        # generation layer
        ans = rag.answer(q)
        records.append({
            "id": r["id"],
            "question": q,
            "answer": ans.spoken_text,
            "ok": ans.ok,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "cited_ids": ans.citations,
            "ground_truth": r["reference_answer"],
            "expected_ids": r["expected_citations"],
        })
        print(f"  [{r['id']}] ok={ans.ok} retrieved={retrieved_ids} "
              f"cited={ans.citations}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# 2. Deterministic retrieval metrics (no LLM)
# ---------------------------------------------------------------------------
def retrieval_metrics(records: list[dict], k: int) -> dict:
    hit, cite_hit, rr = 0, 0, 0.0
    for r in records:
        exp = set(r["expected_ids"])
        got = r["retrieved_ids"]
        if exp & set(got):
            hit += 1
        if exp & set(r["cited_ids"]):
            cite_hit += 1
        rank = next((i + 1 for i, c in enumerate(got) if c in exp), None)
        rr += 1.0 / rank if rank else 0.0
    n = len(records)
    return {
        f"retrieval_hit@{k}": hit / n,
        f"retrieval_mrr@{k}": rr / n,
        "answer_citation_accuracy": cite_hit / n,
        "answer_ok_rate": sum(r["ok"] for r in records) / n,
    }


# ---------------------------------------------------------------------------
# 3. RAGAS metrics (needs judge LLM + embeddings)
# ---------------------------------------------------------------------------
class ChromaONNXEmbeddings:
    """Adapt Chroma's local ONNX MiniLM to the LangChain Embeddings interface
    so RAGAS can use it without pulling torch."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self._ef(list(texts))]

    def embed_query(self, text):
        return list(map(float, self._ef([text])[0]))


def _shim_langchain_vertex() -> None:
    """ragas 0.4.3 hard-imports `langchain_community.chat_models.vertexai`,
    a path removed in the sunset langchain-community 0.4.x. We never use Vertex,
    so register a stub module so the import at ragas load time succeeds."""
    import sys
    import types
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        import langchain_community.chat_models  # noqa: F401
        from langchain_core.language_models.chat_models import BaseChatModel
    except Exception:  # noqa: BLE001
        BaseChatModel = object  # type: ignore
    mod = types.ModuleType(name)
    mod.ChatVertexAI = type("ChatVertexAI", (BaseChatModel,), {})  # type: ignore
    sys.modules[name] = mod


def ragas_scores(records: list[dict], judge_model: str) -> "tuple[dict, object]":
    _shim_langchain_vertex()
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )

    ds = Dataset.from_list([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    } for r in records])

    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "3000")),
    ))
    emb = LangchainEmbeddingsWrapper(ChromaONNXEmbeddings())

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(
            max_workers=int(os.environ.get("JUDGE_WORKERS", "1")),
            timeout=int(os.environ.get("JUDGE_TIMEOUT", "180")),
        ),
    )
    return result._repr_dict if hasattr(result, "_repr_dict") else dict(result), result


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gen-model", default="meta-llama/llama-3.1-8b-instruct",
                    help="model that generates the RAG answers")
    ap.add_argument("--judge", default="openai/gpt-4o-mini",
                    help="RAGAS judge model (routed via OpenRouter)")
    ap.add_argument("--out", default=str(HERE / "ragas_results.json"))
    ap.add_argument("--from-records", metavar="JSON", default=None,
                    help="reuse answers/contexts from a prior run's results "
                         "JSON instead of re-generating (saves API spend)")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set — needed to generate answers "
              "and to run the RAGAS judge.\n  export OPENROUTER_API_KEY=sk-or-...",
              file=sys.stderr)
        sys.exit(2)

    if args.from_records:
        print(f"Reusing records from {args.from_records}", file=sys.stderr)
        records = json.loads(Path(args.from_records).read_text())["records"]
    else:
        print("Running pipeline over golden questions...", file=sys.stderr)
        records = build_records(args.k, args.gen_model)

    det = retrieval_metrics(records, args.k)
    print("\n=== Deterministic retrieval/citation metrics ===")
    for m, v in det.items():
        print(f"  {m:28s} {v:.3f}")

    print("\nRunning RAGAS judge...", file=sys.stderr)
    rag_scores, result = ragas_scores(records, args.judge)
    print("\n=== RAGAS metrics ===")
    for m, v in rag_scores.items():
        try:
            print(f"  {m:28s} {float(v):.3f}")
        except (TypeError, ValueError):
            print(f"  {m:28s} {v}")

    out = {"config": vars(args), "deterministic": det,
           "ragas": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in rag_scores.items()},
           "records": records}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}", file=sys.stderr)

    # per-question CSV for eyeballing
    try:
        df = result.to_pandas()
        df.insert(0, "id", [r["id"] for r in records])
        csv = HERE / "ragas_per_question.csv"
        df.to_csv(csv, index=False)
        print(f"Wrote {csv}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"(per-question CSV skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()

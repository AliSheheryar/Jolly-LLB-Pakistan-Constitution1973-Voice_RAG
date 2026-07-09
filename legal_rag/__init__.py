"""Legal RAG pipeline for a Pakistani-law voice agent (ConversationRelay)."""
from .schema import Chunk, Retrieved, Answer
from .retriever import HybridRetriever
from .vectorstore import ChromaHybridRetriever
from .pipeline import LegalRAG
from .llm import OpenRouterLLM, AnthropicLLM, StubLLM

__all__ = [
    "Chunk", "Retrieved", "Answer",
    "HybridRetriever", "ChromaHybridRetriever", "LegalRAG",
    "OpenRouterLLM", "AnthropicLLM", "StubLLM",
]

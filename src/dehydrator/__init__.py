"""Dehydrator — Client-side BM25 tool search for LLM APIs.

Optionally re-ranks the shortlist with Jev (TypeSafe AI).
"""

from dehydrator._client import AsyncDehydratedClient, DehydratedClient
from dehydrator._index import ToolIndex
from dehydrator._jev import JevIndex, JevReranker, Reranker, detect_provider
from dehydrator._openai_client import (
    AsyncOpenAIDehydratedClient,
    OpenAIDehydratedClient,
)
from dehydrator._types import SearchIndex

__all__ = [
    "AsyncDehydratedClient",
    "AsyncOpenAIDehydratedClient",
    "DehydratedClient",
    "JevIndex",
    "JevReranker",
    "OpenAIDehydratedClient",
    "Reranker",
    "SearchIndex",
    "detect_provider",
    "ToolIndex",
]

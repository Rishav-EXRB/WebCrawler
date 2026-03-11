"""LangGraph state definitions for the crawler pipeline.

State is the interface between the graph and the end user, as well as the
data model used internally by the graph to pass data between nodes.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from crawler.models import (
    CrawledDoc,
    DiscoveredURL,
    ExtractedEntity,
    SearchQuery,
    VerifiedSource,
)


@dataclass(kw_only=True)
class InputState:
    """External API input — everything the caller must provide."""

    user_query: str
    """The raw natural-language query from the user."""
    session_id: str = ""


@dataclass(kw_only=True)
class State(InputState):
    """Full internal state threaded through every node.

    Fields use LangGraph reducers where accumulation is needed.
    """

    # ── Intent Parser output ─────────────────────────────────
    search_queries: list[SearchQuery] = field(default_factory=list)

    # ── URL Discovery output ─────────────────────────────────
    discovered_urls: list[DiscoveredURL] = field(default_factory=list)

    # ── Web Crawler output ───────────────────────────────────
    crawled_docs: list[CrawledDoc] = field(default_factory=list)

    # ── Source Verifier output ───────────────────────────────
    verified_sources: list[VerifiedSource] = field(default_factory=list)

    # ── MongoDB Logger output ────────────────────────────────
    raw_doc_ids: list[str] = field(default_factory=list)
    raw_vector_ids: list[str] = field(default_factory=list)
    session_id: str = ""

    # ── Preprocessor output (Entity Extraction) ──────────────
    extracted_entities: list[ExtractedEntity] = field(default_factory=list)
    entity_vector_ids: list[str] = field(default_factory=list)

    # ── Retry / loop control ─────────────────────────────────
    retry_count: Annotated[int, operator.add] = 0
    max_retries: int = 2

    # ── Cost tracking ────────────────────────────────────────
    cost_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class OutputState:
    """What the caller receives when the graph finishes."""

    extracted_entities: list[ExtractedEntity]
    session_id: str = ""
    raw_doc_ids: list[str] = field(default_factory=list)
    raw_vector_ids: list[str] = field(default_factory=list)
    entity_vector_ids: list[str] = field(default_factory=list)
    cost_summary: dict[str, Any] = field(default_factory=dict)

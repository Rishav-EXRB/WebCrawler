"""LangGraph StateGraph — wires the 6 pipeline nodes with conditional edges.

Flow:
  START → intent_parser → url_discovery →(has URLs?)→ web_crawler
  →(has docs?)→ source_verifier →(has verified?)→ mongo_logger
  → preprocessor →(enough results OR no retries left?)→ END
                  └─(too few & retries left)→ intent_parser (retry)
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph

from crawler.config import Configuration
from crawler.state import InputState, OutputState, State

# ── Import node functions ────────────────────────────────────
from crawler.nodes.intent_parser import parse_intent
from crawler.nodes.url_discovery import discover_urls
from crawler.nodes.web_crawler import crawl_pages
from crawler.nodes.source_verifier import verify_sources
from crawler.nodes.mongo_logger import log_to_mongo
from crawler.nodes.preprocessor import preprocess


# ── Routing functions ────────────────────────────────────────
def route_after_discovery(
    state: State,
) -> Literal["web_crawler", "__end__"]:
    """Skip to END if no URLs were found."""
    if state.discovered_urls:
        return "web_crawler"
    print("[Router] No URLs discovered — ending pipeline.")
    return "__end__"


def route_after_crawl(
    state: State,
) -> Literal["source_verifier", "__end__"]:
    """Skip to END if no documents were successfully crawled."""
    if state.crawled_docs:
        return "source_verifier"
    print("[Router] No documents crawled — ending pipeline.")
    return "__end__"


def route_after_verify(
    state: State,
) -> Literal["mongo_logger", "__end__"]:
    """Skip to END if all sources were rejected."""
    if state.verified_sources:
        return "mongo_logger"
    print("[Router] All sources rejected — ending pipeline.")
    return "__end__"


def route_after_preprocess(
    state: State,
) -> Literal["__end__", "intent_parser"]:
    """Retry if too few results and retries remain, otherwise END."""
    min_docs = state.max_retries  # use max_retries as rough minimum threshold
    # If we have a Configuration with min_processed_docs, prefer that
    # For simplicity, use 3 as default minimum
    min_docs = 3

    if len(state.extracted_entities) >= min_docs:
        print(
            f"[Router] {len(state.extracted_entities)} entities extracted — pipeline complete."
        )
        return "__end__"

    if state.retry_count < state.max_retries:
        print(
            f"[Router] Only {len(state.extracted_entities)} entities "
            f"(need {min_docs}) — retrying ({state.retry_count + 1}/{state.max_retries})."
        )
        return "intent_parser"

    print(
        f"[Router] Only {len(state.extracted_entities)} entities but "
        f"retries exhausted — ending pipeline."
    )
    return "__end__"


# ── Build the graph ──────────────────────────────────────────
workflow = StateGraph(
    State,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)

# Add nodes
workflow.add_node("intent_parser", parse_intent)
workflow.add_node("url_discovery", discover_urls)
workflow.add_node("web_crawler", crawl_pages)
workflow.add_node("source_verifier", verify_sources)
workflow.add_node("mongo_logger", log_to_mongo)
workflow.add_node("preprocessor", preprocess)

# Wire edges
workflow.add_edge("__start__", "intent_parser")
workflow.add_edge("intent_parser", "url_discovery")
workflow.add_conditional_edges("url_discovery", route_after_discovery)
workflow.add_conditional_edges("web_crawler", route_after_crawl)
workflow.add_conditional_edges("source_verifier", route_after_verify)
workflow.add_edge("mongo_logger", "preprocessor")
workflow.add_conditional_edges("preprocessor", route_after_preprocess)

# Compile
graph = workflow.compile()
graph.name = "CrawlerPipeline"

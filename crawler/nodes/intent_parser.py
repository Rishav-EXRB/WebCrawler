"""Intent Parser node — extracts structured search queries from user input.

Uses the Replicate LLM to analyse the user query and produce a list of
SearchQuery objects with topics, preferences, priorities, and diverse
search strings for maximum recall.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import replicate
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import SearchQuery
from crawler.state import State

_SYSTEM_PROMPT = """\
You are a search-query generation expert. Given a user's research request,
produce a JSON array of search queries to send to a web search API.

Each element MUST be a JSON object with these keys:
- "query"       : the search string (be specific, diverse, use different phrasings)
- "topic"       : high-level topic this query relates to
- "preferences" : list of user-stated preferences (e.g. ["recent", "peer-reviewed"])
- "priority"    : "low", "medium", or "high"

Generate 3-5 queries that cover different angles of the user's request.
Return ONLY the JSON array, no markdown fences, no commentary.
"""


async def parse_intent(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Analyse the user query and generate structured SearchQuery objects."""
    configuration = Configuration.from_runnable_config(config)

    user_msg = state.user_query
    if state.retry_count > 0:
        user_msg += (
            "\n\n[RETRY NOTE: Previous search yielded too few results. "
            "Try broader or alternative queries.]"
        )

    t0 = time.time()

    output = replicate.run(
        configuration.model,
        input={
            "prompt": f"{_SYSTEM_PROMPT}\n\nUser request:\n{user_msg}",
            "max_tokens": 1024,
            "temperature": 0.7,
        },
    )

    # Replicate streaming returns an iterator of string chunks
    raw_text = "".join(str(chunk) for chunk in output)
    latency = time.time() - t0

    # Estimate tokens (rough: ~4 chars per token)
    input_tokens = len(f"{_SYSTEM_PROMPT}\n\nUser request:\n{user_msg}") // 4
    output_tokens = len(raw_text) // 4

    tracker.record(
        node="intent_parser",
        model=configuration.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency,
    )

    # Parse LLM response into SearchQuery objects
    try:
        # Strip potential markdown fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: create a single generic query
        parsed = [
            {
                "query": state.user_query,
                "topic": state.user_query,
                "preferences": [],
                "priority": "high",
            }
        ]

    queries = [SearchQuery(**item) for item in parsed]
    print(f"[Intent Parser] Generated {len(queries)} search queries")
    for q in queries:
        print(f"  -> {q.query} (priority={q.priority})")

    return {"search_queries": queries}

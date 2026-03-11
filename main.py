"""Entry point for the LangGraph Crawler Pipeline.

Usage:
    uv run python main.py
    uv run python main.py "your custom query here"
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()  # Load .env before anything else


async def run(query: str) -> None:
    """Invoke the crawler pipeline with the given query."""
    from crawler.graph import graph

    print(f"\n{'=' * 60}")
    print("  LangGraph Crawler Pipeline")
    print(f"  Query: {query}")
    print(f"{'=' * 60}\n")

    result = await graph.ainvoke({"user_query": query})

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {len(result.get('extracted_entities', []))} extracted entities")
    print(f"{'=' * 60}\n")

    for i, entity in enumerate(result.get("extracted_entities", []), 1):
        print(f"-- Entity {i} --------------------------------------")
        print(f"  Name:        {entity.name}")
        print(f"  URL:         {entity.source_url}")
        print(f"  Description: {entity.description[:150]}...")
        if entity.metrics:
            print("  Metrics:")
            for k, v in entity.metrics.items():
                print(f"    - {k}: {v}")
        print(f"  Priority:    {entity.priority_score:.2f}")
        print()

    # Print cost summary
    cost = result.get("cost_summary", {})
    if cost:
        print(f"Total estimated cost: ${cost.get('total_cost_usd', 0):.6f}")
        print(f"Total LLM calls:     {cost.get('total_calls', 0)}")
        print(f"Total tokens:        {cost.get('total_tokens', 0)}")


def main() -> None:
    """Parse args and run the pipeline."""
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Latest advancements in AI agents and multi-agent systems 2025"
    )
    asyncio.run(run(query))


if __name__ == "__main__":
    main()

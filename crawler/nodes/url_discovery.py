"""URL Discovery node — searches the web and gathers candidate URLs.

Uses Tavily Search API to find relevant pages for each SearchQuery,
then deduplicates and filters the results.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

from crawler.config import Configuration
from crawler.models import DiscoveredURL
from crawler.state import State


async def discover_urls(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Search the web for each query and collect unique URLs."""
    configuration = Configuration.from_runnable_config(config)

    tavily = TavilySearch(max_results=configuration.max_search_results)

    seen_urls: set[str] = set()
    urls: list[DiscoveredURL] = []

    for sq in state.search_queries:
        try:
            results = await tavily.ainvoke({"query": sq.query})
        except Exception as exc:
            print(f"[URL Discovery] Tavily error for '{sq.query}': {exc}")
            continue

        if isinstance(results, list):
            items = results
        elif isinstance(results, dict):
            items = results.get("results", [results])
        else:
            items = []

        for item in items:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            urls.append(
                DiscoveredURL(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("content", "")[:500],
                    search_query=sq.query,
                )
            )

    print(
        f"[URL Discovery] Found {len(urls)} unique URLs from {len(state.search_queries)} queries"
    )
    return {"discovered_urls": urls}

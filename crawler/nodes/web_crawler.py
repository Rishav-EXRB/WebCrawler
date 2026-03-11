"""Web Crawler node — fetches page content using crawl4ai with httpx fallback.

Crawls all discovered URLs in parallel using crawl4ai's AsyncWebCrawler.
If a page fails, falls back to a simple httpx GET.  Applies a word-count
quality gate to filter out thin pages.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
from crawl4ai import AsyncWebCrawler
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.models import CrawledDoc
from crawler.state import State


async def _crawl_single(url: str, min_words: int) -> CrawledDoc | None:
    """Try crawl4ai first, then httpx, then give up."""
    # ── Attempt 1: crawl4ai ─────────────────────────────────
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            text = result.markdown or result.extracted_content or ""
            word_count = len(text.split())
            if word_count >= min_words:
                return CrawledDoc(
                    url=url,
                    content=text,
                    word_count=word_count,
                    crawl_method="crawl4ai",
                )
    except Exception as exc:
        safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
        print(f"[Web Crawler] crawl4ai failed for {url}: {safe_exc}")

    # ── Attempt 2: httpx fallback ───────────────────────────
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
            word_count = len(text.split())
            if word_count >= min_words:
                return CrawledDoc(
                    url=url,
                    content=text,
                    word_count=word_count,
                    crawl_method="httpx",
                )
    except Exception as exc:
        safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
        print(f"[Web Crawler] httpx fallback failed for {url}: {safe_exc}")

    return None


async def crawl_pages(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Crawl all discovered URLs in parallel with a quality gate."""
    configuration = Configuration.from_runnable_config(config)
    min_words = configuration.min_word_count

    # Launch all crawls concurrently (with a semaphore to be polite)
    sem = asyncio.Semaphore(5)

    async def _bounded(url: str) -> CrawledDoc | None:
        async with sem:
            return await _crawl_single(url, min_words)

    tasks = [_bounded(u.url) for u in state.discovered_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    docs: list[CrawledDoc] = []
    for r in results:
        if isinstance(r, CrawledDoc):
            docs.append(r)

    print(
        f"[Web Crawler] Crawled {len(docs)} pages "
        f"(of {len(state.discovered_urls)} attempted, "
        f"min_words={min_words})"
    )
    return {"crawled_docs": docs}

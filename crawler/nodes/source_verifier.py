"""Source Verifier node — scores credibility and relevance of crawled docs.

Uses a hardcoded trusted-source registry for quick checks, then calls the
Replicate LLM to score each document on credibility (0-1) and relevance (0-1).
Documents below the minimum credibility threshold are filtered out.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional
from urllib.parse import urlparse

import replicate
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import VerifiedSource
from crawler.state import State

# ── Trusted source registry ─────────────────────────────────
TRUSTED_DOMAINS: set[str] = {
    # Government & education
    ".gov",
    ".edu",
    ".ac.uk",
    ".gov.uk",
    # Major wire services & outlets
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nature.com",
    "science.org",
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    # Tech authorities
    "github.com",
    "stackoverflow.com",
}


def _is_trusted_domain(url: str) -> bool:
    """Check if the URL belongs to a trusted domain."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    for td in TRUSTED_DOMAINS:
        if td.startswith("."):
            if host.endswith(td):
                return True
        elif host == td or host.endswith(f".{td}"):
            return True
    return False


_VERIFY_PROMPT = """\
You are a source credibility evaluator. Given a URL and the first 2000 characters
of its content, return a JSON object with exactly these keys:
- "credibility_score": float 0.0-1.0 (how trustworthy is this source?)
- "relevance_score": float 0.0-1.0 (how relevant is this to the user query?)

User query: {query}
URL: {url}

Content (truncated):
{content}

Return ONLY the JSON object, no markdown, no commentary.
"""


async def verify_sources(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Score and filter crawled documents by credibility and relevance."""
    configuration = Configuration.from_runnable_config(config)

    verified: list[VerifiedSource] = []

    for doc in state.crawled_docs:
        is_trusted = _is_trusted_domain(doc.url)

        # LLM scoring
        prompt = _VERIFY_PROMPT.format(
            query=state.user_query,
            url=doc.url,
            content=doc.content[:2000],
        )

        t0 = time.time()
        try:
            output = replicate.run(
                configuration.model,
                input={
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
            )
            raw_text = "".join(str(chunk) for chunk in output)
            latency = time.time() - t0

            # Track cost
            input_tokens = len(prompt) // 4
            output_tokens = len(raw_text) // 4
            tracker.record(
                node="source_verifier",
                model=configuration.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency,
            )

            # Parse scores
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            scores = json.loads(cleaned)
            cred = float(scores.get("credibility_score", 0.5))
            rel = float(scores.get("relevance_score", 0.5))
        except Exception as exc:
            print(f"[Source Verifier] LLM scoring failed for {doc.url}: {exc}")
            # Fallback: give trusted domains higher default scores
            cred = 0.8 if is_trusted else 0.4
            rel = 0.5

        # Trusted-domain boost
        if is_trusted:
            cred = min(1.0, cred + 0.15)

        if cred >= configuration.min_credibility:
            verified.append(
                VerifiedSource(
                    url=doc.url,
                    content=doc.content,
                    credibility_score=round(cred, 3),
                    relevance_score=round(rel, 3),
                    is_trusted=is_trusted,
                )
            )

    print(
        f"[Source Verifier] {len(verified)} of {len(state.crawled_docs)} "
        f"docs passed (min_credibility={configuration.min_credibility})"
    )
    return {"verified_sources": verified}

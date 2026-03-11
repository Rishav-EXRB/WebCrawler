"""MongoDB Logger node — persists verified documents to MongoDB.

Uses motor (async pymongo) to upsert documents into the `raw_documents`
collection.  Maintains a session document for tracking and prevents
duplicates via URL-keyed upsert.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient

from crawler.config import Configuration
from crawler.state import State

_client: AsyncIOMotorClient | None = None
_chroma_kb_cache: dict[tuple[str, str, int], Any] = {}


def _get_client() -> AsyncIOMotorClient:
    """Lazy-initialise the motor client."""
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(uri)
    return _client


def _get_chroma_kb(configuration: Configuration):
    """Lazy-initialise a Chroma knowledge-base client for verified sources."""
    if not configuration.enable_chroma_sink:
        return None

    key = (
        configuration.chroma_persist_dir,
        configuration.chroma_raw_collection,
        configuration.chroma_embedding_dim,
    )
    kb = _chroma_kb_cache.get(key)
    if kb is not None:
        return kb

    try:
        from crawler.vector import ChromaKnowledgeBase
    except ModuleNotFoundError as exc:
        if exc.name == "chromadb":
            raise RuntimeError(
                "Chroma sink is enabled but 'chromadb' is not installed. "
                "Install it with: pip install -r requirements-kb.txt"
            ) from exc
        raise

    kb = ChromaKnowledgeBase(
        persist_dir=configuration.chroma_persist_dir,
        collection_name=configuration.chroma_raw_collection,
        embedding_dimensions=configuration.chroma_embedding_dim,
    )
    _chroma_kb_cache[key] = kb
    return kb


async def log_to_mongo(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Upsert verified documents into MongoDB and Chroma, then track session."""
    configuration = Configuration.from_runnable_config(config)

    client = _get_client()
    db = client[configuration.mongo_db_name]
    raw_col = db["raw_documents"]
    session_col = db["sessions"]

    now = datetime.now(timezone.utc)

    # ── Create / update session ────────────────────────────
    session_id = state.session_id
    if not session_id:
        result = await session_col.insert_one(
            {
                "user_query": state.user_query,
                "status": "crawling",
                "created_at": now,
                "updated_at": now,
                "retry_count": state.retry_count,
            }
        )
        session_id = str(result.inserted_id)
    else:
        from bson import ObjectId

        await session_col.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"status": "crawling", "updated_at": now}},
        )

    async def _upsert_mongo_docs() -> list[str]:
        doc_ids: list[str] = []
        for src in state.verified_sources:
            result = await raw_col.update_one(
                {"url": src.url},
                {
                    "$set": {
                        "url": src.url,
                        "content": src.content,
                        "credibility_score": src.credibility_score,
                        "relevance_score": src.relevance_score,
                        "is_trusted": src.is_trusted,
                        "session_id": session_id,
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            doc_ids.append(str(result.upserted_id) if result.upserted_id else src.url)
        return doc_ids

    async def _upsert_chroma_docs() -> list[str]:
        kb = _get_chroma_kb(configuration)
        if kb is None:
            return []
        return await asyncio.to_thread(
            kb.upsert_verified_sources,
            state.verified_sources,
            session_id=session_id,
            user_query=state.user_query,
        )

    if configuration.enable_chroma_sink:
        raw_doc_ids, raw_vector_ids = await asyncio.gather(
            _upsert_mongo_docs(),
            _upsert_chroma_docs(),
        )
    else:
        raw_doc_ids = await _upsert_mongo_docs()
        raw_vector_ids = []

    print(
        "[MongoDB Logger] Stored "
        f"{len(raw_doc_ids)} docs in MongoDB and {len(raw_vector_ids)} vectors in Chroma."
    )

    return {
        "raw_doc_ids": raw_doc_ids,
        "raw_vector_ids": raw_vector_ids,
        "session_id": session_id,
    }

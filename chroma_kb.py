"""CLI utility to build/query a ChromaDB knowledge base from crawler outputs.

Examples:
  python chroma_kb.py ingest
  python chroma_kb.py ingest --session-id <mongo_session_id>
  python chroma_kb.py query "top AI incubators in India"
  python chroma_kb.py peek --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any


def _load_kb_class():
    try:
        from crawler.vector import ChromaKnowledgeBase
    except ModuleNotFoundError as exc:
        if exc.name == "chromadb":
            raise SystemExit(
                "Missing dependency 'chromadb'. Install it with: "
                "pip install -r requirements-kb.txt"
            ) from exc
        raise
    return ChromaKnowledgeBase


def _normalize_mongo_doc(doc: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(doc)
    if "_id" in normalized:
        normalized["_id"] = str(normalized["_id"])

    for key in ("created_at", "updated_at"):
        value = normalized.get(key)
        if isinstance(value, datetime):
            normalized[key] = value.isoformat()
    return normalized


async def fetch_extracted_entities(
    *,
    mongo_uri: str,
    mongo_db: str,
    session_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'motor'. Install project deps with: "
            "pip install -r requirements.txt"
        ) from exc

    client = AsyncIOMotorClient(mongo_uri)
    try:
        collection = client[mongo_db]["extracted_entities"]
        query: dict[str, Any] = {}
        if session_id:
            query["session_id"] = session_id

        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_normalize_mongo_doc(doc) for doc in docs]
    finally:
        client.close()


def cmd_ingest(args: argparse.Namespace) -> None:
    ChromaKnowledgeBase = _load_kb_class()

    entities = asyncio.run(
        fetch_extracted_entities(
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
            session_id=args.session_id,
            limit=args.limit,
        )
    )
    if not entities:
        print("No records found in MongoDB extracted_entities collection.")
        return

    kb = ChromaKnowledgeBase(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_dimensions=args.embedding_dim,
    )
    written = kb.upsert_entities(entities)
    print(f"Indexed {written} entities into Chroma collection '{args.collection}'.")
    print(f"Collection size: {kb.count()}")


def cmd_query(args: argparse.Namespace) -> None:
    ChromaKnowledgeBase = _load_kb_class()

    kb = ChromaKnowledgeBase(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_dimensions=args.embedding_dim,
    )
    results = kb.query(
        query_text=args.query_text,
        top_k=args.top_k,
        session_id=args.session_id,
    )
    print(json.dumps(results, indent=2, ensure_ascii=True))


def cmd_peek(args: argparse.Namespace) -> None:
    ChromaKnowledgeBase = _load_kb_class()

    kb = ChromaKnowledgeBase(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_dimensions=args.embedding_dim,
    )
    payload = {
        "collection": args.collection,
        "count": kb.count(),
        "records": kb.peek(limit=args.limit),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChromaDB knowledge base utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--persist-dir",
            default="./chroma_db",
            help="Directory for persistent Chroma data.",
        )
        subparser.add_argument(
            "--collection",
            default="crawler_kb",
            help="Chroma collection name.",
        )
        subparser.add_argument(
            "--embedding-dim",
            type=int,
            default=384,
            help="Dimension used by local hash embedding.",
        )

    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest extracted entities from MongoDB into Chroma.",
    )
    ingest.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB connection string.",
    )
    ingest.add_argument(
        "--mongo-db",
        default=os.getenv("MONGO_DB_NAME", "langgraph_crawler"),
        help="MongoDB database name.",
    )
    ingest.add_argument(
        "--session-id",
        default=None,
        help="Optional session filter; ingest only one crawler session.",
    )
    ingest.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum entities to ingest from MongoDB.",
    )
    add_shared_options(ingest)
    ingest.set_defaults(func=cmd_ingest)

    query = subparsers.add_parser(
        "query",
        help="Run semantic query against Chroma knowledge base.",
    )
    query.add_argument("query_text", help="Search question/text.")
    query.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest results to return.",
    )
    query.add_argument(
        "--session-id",
        default=None,
        help="Optional session filter for semantic search.",
    )
    add_shared_options(query)
    query.set_defaults(func=cmd_query)

    peek = subparsers.add_parser(
        "peek",
        help="Inspect a few stored records from Chroma.",
    )
    peek.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of stored records to print.",
    )
    add_shared_options(peek)
    peek.set_defaults(func=cmd_peek)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

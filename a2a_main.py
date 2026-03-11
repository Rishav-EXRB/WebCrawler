"""CLI entrypoint for the agent-to-agent crawl validation pipeline.

Example:
  python a2a_main.py \
    --query "Top startup incubators in India" \
    --metrics "Funding Amount,Location,Equity Taken"
"""

from __future__ import annotations

import argparse
import asyncio
import json

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional for CLI ergonomics
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _parse_metrics(raw_metrics: str) -> list[str]:
    return [item.strip() for item in raw_metrics.split(",") if item.strip()]


async def _run_pipeline(args: argparse.Namespace) -> None:
    from crawler.agents import AgentToAgentPipeline

    pipeline = AgentToAgentPipeline(
        max_rounds=args.max_rounds,
        chroma_persist_dir=args.chroma_persist_dir,
        chroma_raw_collection=args.chroma_raw_collection,
        chroma_entity_collection=args.chroma_entity_collection,
        chroma_embedding_dim=args.embedding_dim,
    )
    result = await pipeline.run(
        query=args.query,
        required_metrics=_parse_metrics(args.metrics),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-to-agent crawler pipeline")
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language research query.",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated required metrics, e.g. 'Funding Amount,Location'.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum crawl-validation rounds before returning no data available.",
    )
    parser.add_argument(
        "--chroma-persist-dir",
        default="./chroma_db",
        help="Directory for Chroma persistent storage.",
    )
    parser.add_argument(
        "--chroma-raw-collection",
        default="crawler_raw_sources",
        help="Chroma collection for verified source documents.",
    )
    parser.add_argument(
        "--chroma-entity-collection",
        default="crawler_entities",
        help="Chroma collection for extracted entities.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Embedding dimension used by local hash embeddings.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run_pipeline(args))


if __name__ == "__main__":
    main()

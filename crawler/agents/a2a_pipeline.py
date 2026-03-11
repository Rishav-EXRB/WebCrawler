"""Agent-to-agent pipeline for crawl, validate, and targeted recrawl.

Flow:
1) Crawler agent runs pipeline and stores data in MongoDB + ChromaDB.
2) Validator agent reads entity vectors from ChromaDB and checks required metrics.
3) If metrics are missing, validator asks crawler agent for targeted recrawl.
4) If data is still unavailable, return the strict terminal response: "no data available".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from crawler.graph import graph
from crawler.vector import ChromaKnowledgeBase


def _normalize_metric(value: str) -> str:
    return " ".join(value.strip().lower().split())


@dataclass(kw_only=True)
class AgentMessage:
    round_number: int
    from_agent: str
    to_agent: str
    content: str


@dataclass(kw_only=True)
class ValidationOutcome:
    sufficient: bool
    no_data_available: bool
    available_metrics: list[str] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    entity_count: int = 0


@dataclass(kw_only=True)
class CrawlOutcome:
    session_id: str
    entities: list[dict[str, Any]]
    raw_vector_ids: list[str] = field(default_factory=list)
    entity_vector_ids: list[str] = field(default_factory=list)
    cost_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class AgentToAgentResult:
    status: str
    message: str
    session_id: str
    query: str
    required_metrics: list[str]
    available_metrics: list[str] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    communication_log: list[AgentMessage] = field(default_factory=list)
    rounds_used: int = 0
    cost_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "session_id": self.session_id,
            "query": self.query,
            "required_metrics": self.required_metrics,
            "available_metrics": self.available_metrics,
            "missing_metrics": self.missing_metrics,
            "entities": self.entities,
            "communication_log": [asdict(msg) for msg in self.communication_log],
            "rounds_used": self.rounds_used,
            "cost_summary": self.cost_summary,
        }


class CrawlerAgent:
    """Agent 1: crawls data and writes to storage layers."""

    def __init__(
        self,
        *,
        chroma_persist_dir: str = "./chroma_db",
        chroma_raw_collection: str = "crawler_raw_sources",
        chroma_entity_collection: str = "crawler_entities",
        chroma_embedding_dim: int = 384,
    ) -> None:
        self.chroma_persist_dir = chroma_persist_dir
        self.chroma_raw_collection = chroma_raw_collection
        self.chroma_entity_collection = chroma_entity_collection
        self.chroma_embedding_dim = chroma_embedding_dim

    async def crawl(
        self,
        *,
        base_query: str,
        missing_metrics: list[str],
        session_id: str | None,
    ) -> CrawlOutcome:
        if missing_metrics:
            focused = ", ".join(missing_metrics)
            query = (
                f"{base_query}. Focus only on finding explicit values for these metrics: "
                f"{focused}. Return entities with those metrics."
            )
        else:
            query = base_query

        input_payload: dict[str, Any] = {"user_query": query}
        if session_id:
            input_payload["session_id"] = session_id

        result = await graph.ainvoke(
            input_payload,
            config={
                "configurable": {
                    "max_retries": 0,
                    "enable_chroma_sink": True,
                    "chroma_persist_dir": self.chroma_persist_dir,
                    "chroma_raw_collection": self.chroma_raw_collection,
                    "chroma_entity_collection": self.chroma_entity_collection,
                    "chroma_embedding_dim": self.chroma_embedding_dim,
                }
            },
        )

        entities: list[dict[str, Any]] = []
        for item in result.get("extracted_entities", []):
            if hasattr(item, "model_dump"):
                entities.append(item.model_dump())
            elif isinstance(item, dict):
                entities.append(dict(item))

        return CrawlOutcome(
            session_id=str(result.get("session_id") or session_id or ""),
            entities=entities,
            raw_vector_ids=list(result.get("raw_vector_ids") or []),
            entity_vector_ids=list(result.get("entity_vector_ids") or []),
            cost_summary=dict(result.get("cost_summary") or {}),
        )


class ValidatorAgent:
    """Agent 2: validates metric sufficiency using the vector database."""

    def __init__(
        self,
        *,
        chroma_persist_dir: str = "./chroma_db",
        chroma_entity_collection: str = "crawler_entities",
        chroma_embedding_dim: int = 384,
        max_scan_records: int = 1000,
    ) -> None:
        self.kb = ChromaKnowledgeBase(
            persist_dir=chroma_persist_dir,
            collection_name=chroma_entity_collection,
            embedding_dimensions=chroma_embedding_dim,
        )
        self.max_scan_records = max_scan_records

    def validate(
        self,
        *,
        session_id: str,
        required_metrics: list[str],
    ) -> ValidationOutcome:
        if not session_id:
            return ValidationOutcome(
                sufficient=False,
                no_data_available=True,
                missing_metrics=required_metrics,
                available_metrics=[],
                entity_count=0,
            )

        records = self.kb.get_records(
            where={"session_id": session_id},
            limit=self.max_scan_records,
        )

        entity_records = [
            record
            for record in records
            if (record.get("metadata") or {}).get("record_type") == "entity"
        ]
        if not entity_records:
            return ValidationOutcome(
                sufficient=False,
                no_data_available=True,
                missing_metrics=required_metrics,
                available_metrics=[],
                entity_count=0,
            )

        available_norm: set[str] = set()
        for record in entity_records:
            metadata = record.get("metadata") or {}
            metric_keys_csv = str(metadata.get("metric_keys_csv", "")).strip()
            if not metric_keys_csv:
                continue
            for raw_key in metric_keys_csv.split(","):
                norm = _normalize_metric(raw_key)
                if norm:
                    available_norm.add(norm)

        required_map: dict[str, str] = {}
        for metric in required_metrics:
            norm = _normalize_metric(metric)
            if norm and norm not in required_map:
                required_map[norm] = metric.strip()

        missing = [
            original
            for norm, original in required_map.items()
            if norm not in available_norm
        ]
        available = sorted(available_norm)

        return ValidationOutcome(
            sufficient=len(missing) == 0,
            no_data_available=False,
            available_metrics=available,
            missing_metrics=missing,
            entity_count=len(entity_records),
        )


class AgentToAgentPipeline:
    """Coordinates crawler and validator agents with strict no-fallback behavior."""

    def __init__(
        self,
        *,
        max_rounds: int = 3,
        chroma_persist_dir: str = "./chroma_db",
        chroma_raw_collection: str = "crawler_raw_sources",
        chroma_entity_collection: str = "crawler_entities",
        chroma_embedding_dim: int = 384,
    ) -> None:
        self.max_rounds = max_rounds
        self.crawler_agent = CrawlerAgent(
            chroma_persist_dir=chroma_persist_dir,
            chroma_raw_collection=chroma_raw_collection,
            chroma_entity_collection=chroma_entity_collection,
            chroma_embedding_dim=chroma_embedding_dim,
        )
        self.validator_agent = ValidatorAgent(
            chroma_persist_dir=chroma_persist_dir,
            chroma_entity_collection=chroma_entity_collection,
            chroma_embedding_dim=chroma_embedding_dim,
        )

    async def run(
        self,
        *,
        query: str,
        required_metrics: list[str],
    ) -> AgentToAgentResult:
        normalized_required = [metric.strip() for metric in required_metrics if metric.strip()]
        if not normalized_required:
            return AgentToAgentResult(
                status="no_data_available",
                message="no data available",
                session_id="",
                query=query,
                required_metrics=[],
            )

        session_id = ""
        missing_metrics = list(normalized_required)
        communication_log: list[AgentMessage] = []
        latest_entities: list[dict[str, Any]] = []
        latest_cost_summary: dict[str, Any] = {}

        for round_number in range(1, self.max_rounds + 1):
            communication_log.append(
                AgentMessage(
                    round_number=round_number,
                    from_agent="validator_agent",
                    to_agent="crawler_agent",
                    content=(
                        "Crawl request: "
                        f"fetch data for missing metrics [{', '.join(missing_metrics)}]."
                    ),
                )
            )

            crawl_outcome = await self.crawler_agent.crawl(
                base_query=query,
                missing_metrics=missing_metrics,
                session_id=session_id or None,
            )
            session_id = crawl_outcome.session_id or session_id
            latest_entities = crawl_outcome.entities
            latest_cost_summary = crawl_outcome.cost_summary

            communication_log.append(
                AgentMessage(
                    round_number=round_number,
                    from_agent="crawler_agent",
                    to_agent="validator_agent",
                    content=(
                        f"Crawl response: session_id={session_id or 'unknown'}, "
                        f"entities={len(crawl_outcome.entities)}, "
                        f"entity_vectors={len(crawl_outcome.entity_vector_ids)}."
                    ),
                )
            )

            validation = self.validator_agent.validate(
                session_id=session_id,
                required_metrics=normalized_required,
            )

            if validation.no_data_available:
                communication_log.append(
                    AgentMessage(
                        round_number=round_number,
                        from_agent="validator_agent",
                        to_agent="orchestrator",
                        content="no data available",
                    )
                )
                return AgentToAgentResult(
                    status="no_data_available",
                    message="no data available",
                    session_id=session_id,
                    query=query,
                    required_metrics=normalized_required,
                    available_metrics=validation.available_metrics,
                    missing_metrics=validation.missing_metrics,
                    entities=latest_entities,
                    communication_log=communication_log,
                    rounds_used=round_number,
                    cost_summary=latest_cost_summary,
                )

            if validation.sufficient:
                communication_log.append(
                    AgentMessage(
                        round_number=round_number,
                        from_agent="validator_agent",
                        to_agent="orchestrator",
                        content="Sufficient data available in vector db.",
                    )
                )
                return AgentToAgentResult(
                    status="sufficient",
                    message="sufficient data available",
                    session_id=session_id,
                    query=query,
                    required_metrics=normalized_required,
                    available_metrics=validation.available_metrics,
                    missing_metrics=[],
                    entities=latest_entities,
                    communication_log=communication_log,
                    rounds_used=round_number,
                    cost_summary=latest_cost_summary,
                )

            missing_metrics = validation.missing_metrics
            communication_log.append(
                AgentMessage(
                    round_number=round_number,
                    from_agent="validator_agent",
                    to_agent="crawler_agent",
                    content=(
                        "Insufficient data. Recrawl required for metrics: "
                        f"[{', '.join(missing_metrics)}]."
                    ),
                )
            )

        communication_log.append(
            AgentMessage(
                round_number=self.max_rounds,
                from_agent="validator_agent",
                to_agent="orchestrator",
                content="no data available",
            )
        )
        return AgentToAgentResult(
            status="no_data_available",
            message="no data available",
            session_id=session_id,
            query=query,
            required_metrics=normalized_required,
            missing_metrics=missing_metrics,
            entities=latest_entities,
            communication_log=communication_log,
            rounds_used=self.max_rounds,
            cost_summary=latest_cost_summary,
        )

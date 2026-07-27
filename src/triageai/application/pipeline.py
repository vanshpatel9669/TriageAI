"""Wires the five nodes into a LangGraph StateGraph and exposes a single
`TriagePipeline.run()` entrypoint. All dependencies are injected via the
constructor (DI) — this module never imports a concrete infrastructure class.
"""

from __future__ import annotations

import time
from typing import cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from triageai.application.nodes.classification import ClassificationNode
from triageai.application.nodes.drafting import DraftingNode
from triageai.application.nodes.ingestion import IngestionNode
from triageai.application.nodes.retrieval import RetrievalNode
from triageai.application.nodes.routing import RoutingNode
from triageai.application.state import PipelineState
from triageai.domain.interfaces import (
    GraphMemoryRepository,
    LLMProvider,
    TicketRepository,
    VectorSearchRepository,
)
from triageai.domain.models import Ticket, TriageResult


class TriagePipeline:
    def __init__(
        self,
        llm: LLMProvider,
        vector_repo: VectorSearchRepository,
        graph_repo: GraphMemoryRepository,
        ticket_repo: TicketRepository,
        confidence_threshold: float = 0.75,
    ) -> None:
        self._graph_repo = graph_repo
        self._ticket_repo = ticket_repo
        self._graph = self._build_graph(llm, vector_repo, graph_repo, confidence_threshold)

    @staticmethod
    def _build_graph(
        llm: LLMProvider,
        vector_repo: VectorSearchRepository,
        graph_repo: GraphMemoryRepository,
        confidence_threshold: float,
    ) -> CompiledStateGraph:
        builder = StateGraph(PipelineState)
        builder.add_node("ingestion", IngestionNode())
        builder.add_node("classification", ClassificationNode(llm))
        builder.add_node("retrieval", RetrievalNode(vector_repo, graph_repo))
        builder.add_node("drafting", DraftingNode(llm))
        builder.add_node("routing", RoutingNode(confidence_threshold))

        builder.set_entry_point("ingestion")
        builder.add_edge("ingestion", "classification")
        builder.add_edge("classification", "retrieval")
        builder.add_edge("retrieval", "drafting")
        builder.add_edge("drafting", "routing")
        builder.add_edge("routing", END)

        return builder.compile()

    async def run(self, ticket: Ticket) -> TriageResult:
        start = time.monotonic()
        raw_state = await self._graph.ainvoke({"ticket": ticket})
        final_state: PipelineState = cast(PipelineState, raw_state)
        latency_ms = (time.monotonic() - start) * 1000

        result = TriageResult(
            ticket=final_state["ticket"],
            classification=final_state["classification"],
            evidence=final_state.get("evidence", []),
            draft=final_state["draft"],
            routing=final_state["routing"],
            latency_ms=round(latency_ms, 2),
        )

        await self._graph_repo.record_ticket_relationship(result.ticket, result.classification)
        await self._ticket_repo.save(result)
        return result

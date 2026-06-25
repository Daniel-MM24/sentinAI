"""FastAPI application for SentinAI audit API."""
import json
import logging
import os
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.graph import create_audit_graph
from src.retrieval.vector_db import VectorStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SentinAI Audit API",
    description="Autonomous, auditable multi-agent system for financial compliance",
    version="0.1.0",
)


class AuditRequest(BaseModel):
    """Request schema for running an audit."""

    query: str = Field(..., description="The audit query or transaction to evaluate.")
    thread_id: str = Field(
        default=None, description="Optional thread ID for resuming previous audits."
    )


# Initialize global components
_vector_store: VectorStore | None = None
_checkpoint_path: str = os.getenv("CHECKPOINT_DB_PATH", "./checkpoints.sqlite")


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "sentinai_compliance")
        _vector_store = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
    return _vector_store


async def _audit_event_stream(
    graph, request: AuditRequest
) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Events for audit workflow execution."""
    try:
        yield "event: start\ndata: {\"status\": \"started\", \"query\": %s}\n\n" % json.dumps(
            request.query
        )

        async for update in graph.astream(request.query, request.thread_id):
            for node_name, node_state in update.items():
                event_data = {
                    "node": node_name,
                    "status": "completed",
                    "state": {
                        "messages_count": len(node_state.get("messages", [])),
                        "has_evaluation": node_state.get("evaluation") is not None,
                        "has_report": node_state.get("final_report") is not None,
                    },
                }
                yield f"event: node\ndata: {json.dumps(event_data)}\n\n"

        # Final completion event
        yield "event: done\ndata: {\"status\": \"completed\"}\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.exception("Audit stream error: %s", exc)
        error_data = {"status": "error", "error": str(exc)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.post("/api/v1/audit/run")
async def run_audit(request: AuditRequest) -> StreamingResponse:
    """
    Run a compliance audit with streaming updates.

    Returns Server-Sent Events (text/event-stream) for real-time workflow updates.
    """
    try:
        vector_store = get_vector_store()
        graph = create_audit_graph(
            vector_store=vector_store,
            checkpoint_path=_checkpoint_path,
        )

        return StreamingResponse(
            _audit_event_stream(graph, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start audit: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "sentinai-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from ..agent.query import query
from ..prompt.assembler import PromptAssembler
from .state import RuntimeState


class TurnRequest(BaseModel):
    message: str


def create_app(state: RuntimeState) -> FastAPI:
    app = FastAPI(title="agent-reforge Runtime API")

    async def verify(x_api_key: str = Header(default="")) -> None:
        if x_api_key != state.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing api key")

    @app.post("/v1/threads")
    async def create_thread(_: None = Depends(verify)) -> dict[str, str]:
        return {"id": state.create_thread().id}

    @app.post("/v1/threads/{thread_id}/turns")
    async def create_turn(
        thread_id: str, body: TurnRequest, _: None = Depends(verify)
    ) -> dict[str, Any]:
        thread = state.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")

        # build system prompt (memories may have changed since last turn)
        system_prompt = PromptAssembler(
            cwd=state.cwd,
            model=state.client.model_name,
            provider=state.client.provider_name,
            tool_names=state.registry.list_names(),
            memories=[m.content for m in state.memory.list_memory()],
        ).build()

        # run the ReAct loop once; append each event it produces to the thread.
        async for event in query(
            client=state.client,
            registry=state.registry,
            system_prompt=system_prompt,
            user_message=body.message,
            cwd=state.cwd,
            memory=state.memory,
            code_index=state.code_index,
        ):
            thread.events.append(_json_safe(event))
        return {"status": "completed", "event_count": len(thread.events)}

    @app.get("/v1/threads/{thread_id}/events")
    # user get the events, then have the result
    async def get_events(thread_id: str, _: None = Depends(verify)) -> dict[str, Any]:
        thread = state.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return {"thread_id": thread.id, "events": thread.events}

    return app


def _json_safe(event: dict[str, Any]) -> dict[str, Any]:
    # events may carry non-JSON objects (Exception, Message); keep it serializable.
    etype = event.get("type")
    if etype == "error":
        return {"type": "error", "error": str(event.get("error"))}
    if etype == "done":
        return {
            "type": "done",
            "turns": event.get("turns"),
            "total_tokens": event.get("total_tokens"),
        }
    # text_delta / tool_result / usage are already plain dicts of str/int/bool.
    return event

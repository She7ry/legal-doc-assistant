from __future__ import annotations

import json

from fastapi.encoders import jsonable_encoder

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def format_sse(event: str, data: object, *, event_id: int | None = None) -> str:
    encoded = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {encoded}\n\n"

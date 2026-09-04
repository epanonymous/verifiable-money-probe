"""Durably submit and non-destructively poll the deployed derivation app."""

from __future__ import annotations

from typing import Any

APP_NAME = "vmp-exp4-derive"
FUNCTION_NAME = "derive"


def _modal(modal_module: Any | None = None) -> Any:
    if modal_module is not None:
        return modal_module
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError(
            "the durable launcher requires the Modal Python client"
        ) from exc
    return modal


def submit_derivation(which: str, modal_module: Any | None = None) -> str:
    """Spawn one deployed input and return its durable function-call ID."""

    if which not in {"main", "lbr"}:
        raise ValueError("which must be 'main' or 'lbr'")
    modal = _modal(modal_module)
    function_call = modal.Function.from_name(APP_NAME, FUNCTION_NAME).spawn(which)
    call_id = str(function_call.object_id)
    if not call_id:
        raise RuntimeError("Modal returned an empty function-call ID")
    print(call_id)
    return call_id


def poll_derivation(
    call_id: str, timeout: float = 0.0, modal_module: Any | None = None
) -> dict[str, Any]:
    """Poll an existing call without cancelling or otherwise mutating it."""

    if not call_id:
        raise ValueError("call_id must not be empty")
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    modal = _modal(modal_module)
    function_call = modal.FunctionCall.from_id(call_id)
    output_expired = getattr(
        getattr(modal, "exception", None), "OutputExpiredError", None
    )
    try:
        result = function_call.get(timeout=timeout)
    except TimeoutError:
        return {"call_id": call_id, "status": "running"}
    except Exception as exc:
        if output_expired is not None and isinstance(exc, output_expired):
            return {"call_id": call_id, "status": "expired"}
        raise
    return {"call_id": call_id, "status": "completed", "result": result}

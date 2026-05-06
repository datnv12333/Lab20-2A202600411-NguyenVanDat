import json
from pathlib import Path
from typing import Union

from multi_agent_research_lab.core.state import ResearchState


def write_trace(path: Union[str, Path], state: ResearchState) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": state.request.model_dump(),
        "iteration": state.iteration,
        "route_history": state.route_history,
        "errors": state.errors,
        "stopped_reason": state.stopped_reason,
        "agent_results": [r.model_dump() for r in state.agent_results],
        "trace": state.trace,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class GateResult:
    """Standard response object for all gating functions."""
    ok: bool
    code: str
    message: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "code": str(self.code),
            "message": str(self.message),
            "metrics": dict(self.metrics or {}),
        }

def _cfg_get(cfg: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    """Helper to safely retrieve config values."""
    if not cfg:
        return default
    v = cfg.get(key, default)
    return default if v is None else v
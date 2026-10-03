"""AuthZ (RBAC) and rate limiting. Dev uses static tokens; production swaps in OIDC/JWT verification."""
from __future__ import annotations

import os
import time

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ROLES = ("viewer", "researcher", "trader", "risk_officer", "admin")
_bearer = HTTPBearer(auto_error=False)


def _tokens() -> dict[str, str]:
    # API_TOKENS="tok1:admin,tok2:viewer"  (never commit real tokens; load from a secrets manager)
    raw = os.getenv("API_TOKENS", "dev-admin:admin,dev-viewer:viewer")
    return dict(p.split(":", 1) for p in raw.split(",") if ":" in p)


def require(min_role: str):
    need = ROLES.index(min_role)

    def dep(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
        role = _tokens().get(cred.credentials) if cred else None
        if role is None:
            raise HTTPException(401, "missing or invalid token")
        if ROLES.index(role) < need:
            raise HTTPException(403, f"requires {min_role}")
        return role

    return dep


class TokenBucket:
    """Per-client token bucket (in-process; use Redis INCR+EXPIRE when running >1 replica)."""

    def __init__(self, rate: float = 20.0, burst: int = 40) -> None:
        self.rate, self.burst, self.state = rate, burst, {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self.state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1:
            self.state[key] = (tokens, now)
            return False
        self.state[key] = (tokens - 1, now)
        if len(self.state) > 100_000:
            self.state.clear()
        return True


bucket = TokenBucket(float(os.getenv("RATE_LIMIT_RPS", "20")), int(os.getenv("RATE_LIMIT_BURST", "40")))


def client_key(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")

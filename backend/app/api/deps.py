"""FastAPI dependencies: auth, current user."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import verify_token

bearer_scheme = HTTPBearer(auto_error=False)


# P2-11: JWT 改为 HttpOnly cookie, 但继续兼容 Bearer header (API 调用 / 第三方工具).
def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """优先从 Authorization: Bearer 取 token, 否则回退到 HttpOnly cookie.

    这样前端可用 credentials: 'include' 自动带 cookie,
    同时保留对脚本 / 第三方工具的 Bearer 兼容.
    """
    token = ""
    if creds and creds.credentials:
        token = creds.credentials
    if not token:
        token = request.cookies.get("unikb_token") or ""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    sub = verify_token(token)
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return sub

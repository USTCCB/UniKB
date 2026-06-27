"""Health check."""
from datetime import datetime

from fastapi import APIRouter

from app import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "time": datetime.utcnow().isoformat(),
    }

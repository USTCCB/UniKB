"""Centralized logging."""
import sys
from loguru import logger

from app.core.config import settings

logger.remove()
logger.add(
    sys.stdout,
    level=settings.app_log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)

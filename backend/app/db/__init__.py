"""数据库初始化入口."""
from __future__ import annotations

from app.db.base import Base
from app.db.models import User  # noqa: F401
from app.db.session import engine, SessionLocal


def create_tables() -> None:
    """根据模型建表 (应用启动时调用)."""
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    Base.metadata.drop_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ["Base", "User", "engine", "SessionLocal", "create_tables", "drop_tables", "get_db"]

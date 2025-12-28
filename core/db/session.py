import os
import logging
from functools import lru_cache
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def _get_database_url() -> str:
    """
    Primary DB URL for Postgres persistence.
    Example: postgresql+psycopg2://user:pass@host:5432/dbname
    """
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url:
        return url
    # Dev fallback (keeps app runnable without Postgres); prod should set DATABASE_URL.
    fallback = "sqlite:///data/app.db"
    logger.warning("DATABASE_URL not set; using sqlite fallback at %s", fallback)
    return fallback


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = _get_database_url()
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(url, pool_pre_ping=True, future=True, connect_args=connect_args)


@lru_cache(maxsize=1)
def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db_session():
    """
    Get a database session (context manager).
    SQLAlchemy Session objects are context managers, so this works directly.
    Usage:
      with get_db_session() as db:
          ...
          db.commit()
    """
    SessionLocal = get_session_factory()
    return SessionLocal()




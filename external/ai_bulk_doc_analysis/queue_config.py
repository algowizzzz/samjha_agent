"""
Redis Queue configuration for AI Bulk Doc Analysis.
"""
import os
import logging
from redis import Redis
from rq import Queue

logger = logging.getLogger(__name__)

# Global Redis connection and queues
_redis_conn = None
_conversion_queue = None
_execution_queue = None


def get_redis_connection() -> Redis:
    """Get or create Redis connection."""
    global _redis_conn
    if _redis_conn is not None:
        return _redis_conn

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_conn = Redis.from_url(redis_url, decode_responses=False)
        # Test connection
        _redis_conn.ping()
        logger.info(f"Connected to Redis at {redis_url}")
        return _redis_conn
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise


def get_conversion_queue() -> Queue:
    """Get conversion queue (PDF → Markdown)."""
    global _conversion_queue
    if _conversion_queue is None:
        redis_conn = get_redis_connection()
        _conversion_queue = Queue("conversion", connection=redis_conn, default_timeout=300)
    return _conversion_queue


def get_execution_queue() -> Queue:
    """Get execution queue (Claude step execution)."""
    global _execution_queue
    if _execution_queue is None:
        redis_conn = get_redis_connection()
        _execution_queue = Queue("execution", connection=redis_conn, default_timeout=600)
    return _execution_queue


def init_queues():
    """Initialize queues (call at startup)."""
    try:
        get_conversion_queue()
        get_execution_queue()
        logger.info("Queues initialized successfully")
    except Exception as e:
        logger.warning(f"Queue initialization failed (workers may not be running): {e}")


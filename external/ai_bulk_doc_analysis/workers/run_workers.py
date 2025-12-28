#!/usr/bin/env python3
"""
Script to run RQ workers for AI Bulk Doc Analysis.
Usage:
    python run_workers.py          # Run all queues
    python run_workers.py conversion  # Run only conversion queue
    python run_workers.py execution   # Run only execution queue
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rq import Worker, Queue
from redis import Redis

from external.ai_bulk_doc_analysis.queue_config import get_redis_connection

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Get Redis connection
    redis_conn = get_redis_connection()
    
    # Determine which queues to listen to
    if len(sys.argv) > 1:
        queue_names = sys.argv[1:]
    else:
        queue_names = ["conversion", "execution"]
    
    queues = [Queue(name, connection=redis_conn) for name in queue_names]
    
    print(f"Starting RQ worker for queues: {', '.join(queue_names)}")
    print("Press Ctrl+C to stop")
    
    # Create and run worker
    worker = Worker(queues, connection=redis_conn, name='bulk-doc-worker')
    try:
        worker.work(with_scheduler=False)
    except KeyboardInterrupt:
        print("\nWorker stopped by user")


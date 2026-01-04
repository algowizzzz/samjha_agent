#!/usr/bin/env python3
"""
Track progress of a workflow run in real-time
Usage: python track_progress.py [run_id]
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, '.')
from external.ai_bulk_doc_analysis.db_service import get_db_session
from external.ai_bulk_doc_analysis.models import Run as DBRun, StepResult, ExecutionTask

def track_progress(run_id=None):
    """Track progress of a workflow run"""
    
    with get_db_session() as db:
        if run_id:
            run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
        else:
            run = db.query(DBRun).order_by(DBRun.created_at.desc()).first()
        
        if not run:
            print("No run found")
            return
        
        print(f"\n{'='*60}")
        print(f"Tracking Run: {run.run_id}")
        print(f"Status: {run.status}")
        print(f"Created: {run.created_at}")
        print(f"{'='*60}\n")
        
        # Check if CSV workflow
        tasks = db.query(ExecutionTask).filter(ExecutionTask.run_id == run.run_id).all()
        is_csv = len(tasks) > 0
        
        last_status = None
        
        while True:
            # Refresh from DB
            db.expire_all()
            run = db.query(DBRun).filter(DBRun.run_id == run.run_id).first()
            step_results = db.query(StepResult).filter(StepResult.run_id == run.run_id).all()
            
            # Count statuses
            from collections import defaultdict
            status_counts = defaultdict(int)
            for sr in step_results:
                status_counts[sr.status] += 1
            
            success_count = status_counts.get('SUCCESS', 0)
            error_count = status_counts.get('ERROR', 0)
            running_count = status_counts.get('RUNNING', 0)
            queued_count = status_counts.get('QUEUED', 0)
            total_count = len(step_results)
            
            progress_pct = (success_count / total_count * 100) if total_count > 0 else 0
            
            # Only print if status changed
            current_status = f"{run.status}|{success_count}/{total_count}"
            if current_status != last_status:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Status: {run.status} | Progress: {success_count}/{total_count} ({progress_pct:.1f}%)")
                print(f"  ✅ SUCCESS: {success_count} | ⏳ QUEUED: {queued_count} | 🔄 RUNNING: {running_count} | ❌ ERROR: {error_count}")
                
                if is_csv and tasks:
                    # Show task breakdown
                    print(f"  Tasks: {len(tasks)} total")
                    completed_tasks = 0
                    for task in tasks:
                        task_results = [sr for sr in step_results if sr.task_id == task.task_id]
                        task_success = sum(1 for sr in task_results if sr.status == 'SUCCESS')
                        task_total = len(task_results)
                        if task_success == task_total and task_total > 0:
                            completed_tasks += 1
                    print(f"  Completed tasks: {completed_tasks}/{len(tasks)}")
                
                print()
                last_status = current_status
            
            # Check if complete
            if run.status in ['COMPLETE', 'ERROR']:
                print(f"\n{'='*60}")
                print(f"Run {run.status}!")
                print(f"Final: {success_count}/{total_count} steps successful")
                if error_count > 0:
                    print(f"Errors: {error_count} steps failed")
                print(f"{'='*60}\n")
                break
            
            time.sleep(2)

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        track_progress(run_id)
    except KeyboardInterrupt:
        print("\n\nTracking stopped by user")


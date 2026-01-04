"""
Execution Worker - Handles Claude API step execution jobs.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from external.platform.llm.client import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM client not available")

logger = logging.getLogger(__name__)


def execute_step_job(job_data: Dict[str, Any], storage_base: Path):
    """
    RQ job handler for EXECUTE_STEP jobs.
    
    Args:
        job_data: {
            "run_id": str,
            "doc_id": str,
            "session_id": str,  # NEW: Needed to find R0.md
            "step_index": int,
            "task_id": str,  # NEW: Optional, for CSV task-based execution
            "chain_version_id": str,
            "required_inputs": List[str],  # ['R0', 'R1', ...]
            "prompt": str,
            "model_config": {
                "model": str,
                "max_tokens": int,
                "temperature": float
            },
            "idempotency_key": str
        }
        storage_base: Base path for storage
    """
    run_id = job_data["run_id"]
    doc_id = job_data["doc_id"]
    session_id = job_data.get("session_id")  # NEW: For finding R0.md
    step_index = job_data["step_index"]
    task_id = job_data.get("task_id")  # NEW: For CSV workflows
    
    logger.info(f"Starting step {step_index} execution for run {run_id}, doc {doc_id}" + (f", task {task_id}" if task_id else ""))
    
    try:
        # Load required R inputs
        r_inputs = _load_r_inputs(
            storage_base=storage_base,
            run_id=run_id,
            doc_id=doc_id,
            session_id=session_id,  # NEW: Needed to find R0.md
            task_id=task_id,  # NEW
            required_inputs=job_data["required_inputs"]
        )
        
        # Construct prompt with R inputs
        prompt = _construct_prompt(
            base_prompt=job_data["prompt"],
            r_inputs=r_inputs
        )
        
        # Call Claude API
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM client not available")
        
        model_config = job_data.get("model_config", {})
        llm_client = get_llm_client()
        
        # Use model from config if specified
        model = model_config.get("model", "claude-3-haiku-20240307")
        max_tokens = model_config.get("max_tokens", 4096)
        temperature = model_config.get("temperature", 0.2)
        
        # Call Claude API with model config
        # Note: This assumes LLM client supports model override via client.messages.create
        messages = [{"role": "user", "content": prompt}]
        system_prompt = """You are executing a step in a multi-step document analysis chain.
Follow the step instructions precisely and produce the requested output.
The output should be well-formatted and ready for use in subsequent steps."""
        
        import time
        start_time = time.time()
        
        response = llm_client.client.messages.create(
            model=model,
            messages=messages,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract text from response
        output_text = ""
        # Handle different response formats
        if hasattr(response, "content"):
            # Standard format with content blocks
            for block in response.content or []:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str):
                        output_text += block_text
        elif hasattr(response, "text"):
            # Direct text attribute
            output_text = response.text
        elif isinstance(response, str):
            # Already a string
            output_text = response
        
        # Get usage info
        usage = getattr(response, "usage", None)
        
        # Extract token usage
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or 0 if usage else 0
        output_tokens = getattr(usage, "output_tokens", None) or 0 if usage else 0
        
        # Save R(n) output
        r_key = f"R{step_index}"
        if task_id:
            # CSV: save to task-specific directory
            run_dir = storage_base / "runs" / run_id / "docs" / doc_id / "tasks" / task_id
        else:
            run_dir = storage_base / "runs" / run_id / "docs" / doc_id
        run_dir.mkdir(parents=True, exist_ok=True)
        r_path = run_dir / f"{r_key}.md"
        r_path.write_text(output_text, encoding='utf-8')
        
        # Update StepResult in DB
        from external.ai_bulk_doc_analysis.db_service import get_db_session
        from external.ai_bulk_doc_analysis.models import StepResult as DBStepResult
        
        with get_db_session() as db:
            # Find the step result record
            query = db.query(DBStepResult).filter(
                DBStepResult.run_id == run_id,
                DBStepResult.doc_id == doc_id,
                DBStepResult.step_index == step_index
            )
            
            if task_id:
                # For CSV workflows, also filter by task_id
                query = query.filter(DBStepResult.task_id == task_id)
            
            step_result = query.first()
            
            if step_result:
                # Update with results
                step_result.status = "SUCCESS"
                step_result.output_object_key = str(r_path.relative_to(storage_base))
                
                # Add usage info if available
                if usage:
                    step_result.input_tokens = getattr(usage, "input_tokens", None)
                    step_result.output_tokens = getattr(usage, "output_tokens", None)
                
                # Add model info
                if model_config:
                    step_result.model = model_config.get("model")
                    step_result.max_tokens = model_config.get("max_tokens")
                    step_result.temperature = model_config.get("temperature")
                
                # Add latency
                if latency_ms:
                    step_result.latency_ms = latency_ms
                
                # Add request ID if available
                if hasattr(response, "id"):
                    step_result.claude_request_id = response.id
                
                db.commit()
                logger.info(f"Updated StepResult in DB for step {step_index}")
                
                # Check if this is the last step and if conversion is needed
                chain_version_id = job_data.get("chain_version_id")
                _check_and_enqueue_conversion_if_needed(
                    db, run_id, doc_id, task_id, chain_version_id, 
                    step_index, r_path, storage_base
                )
                
                # Check if all steps for this run are complete and update run status
                _update_run_status_if_complete(db, run_id, chain_version_id)
                
                # Check if all steps for this run are complete and update run status
                _update_run_status_if_complete(db, run_id, chain_version_id)
            else:
                logger.warning(f"StepResult not found for run_id={run_id}, doc_id={doc_id}, step_index={step_index}, task_id={task_id}")
        
        logger.info(f"Step {step_index} execution complete for doc {doc_id}")
        return {
            "status": "SUCCESS",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_path": str(r_path),
            "latency_ms": latency_ms,
        }
        
    except Exception as e:
        logger.error(f"Step {step_index} execution failed: {e}", exc_info=True)
        # TODO: Update StepResult status to ERROR in DB
        # db.update_step_result_status(run_id, doc_id, step_index, "ERROR", error_message=str(e))
        raise


def _load_r_inputs(storage_base: Path, run_id: str, doc_id: str, required_inputs: list, session_id: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, str]:
    """Load R0, R1, ... inputs from storage."""
    r_inputs = {}
    
    for r_key in required_inputs:
        # R0 comes from session docs (or task row_data for CSV)
        if r_key == "R0":
            if task_id:
                # CSV workflow: R0 comes from task row_data
                from external.ai_bulk_doc_analysis.db_service import get_db_session
                from external.ai_bulk_doc_analysis.models import ExecutionTask
                
                with get_db_session() as db:
                    task = db.query(ExecutionTask).filter(ExecutionTask.task_id == task_id).first()
                    if not task:
                        raise ValueError(f"Task {task_id} not found")
                    
                    # Format row_data as markdown or JSON based on workflow config
                    # For now, use JSON format
                    import json
                    row_data_str = json.dumps(task.row_data, indent=2)
                    r_inputs[r_key] = f"# CSV Row Data (Row {task.row_index})\n\n```json\n{row_data_str}\n```"
            else:
                # Non-CSV: Find R0 in session docs (under sessions/{session_id}/docs/{doc_id}/R0.md)
                if not session_id:
                    raise ValueError("session_id is required to find R0.md")
                r0_path = storage_base / "sessions" / session_id / "docs" / doc_id / "R0.md"
                if r0_path.exists():
                    r_inputs[r_key] = r0_path.read_text(encoding='utf-8')
                else:
                    raise FileNotFoundError(f"Required input {r_key} not found: {r0_path}")
        else:
            # R1, R2, ... from previous steps
            # For CSV: use task-specific path
            if task_id:
                r_path = storage_base / "runs" / run_id / "docs" / doc_id / "tasks" / task_id / f"{r_key}.md"
            else:
                r_path = storage_base / "runs" / run_id / "docs" / doc_id / f"{r_key}.md"
            
            if r_path.exists():
                r_inputs[r_key] = r_path.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError(f"Required input {r_key} not found: {r_path}")
    
    return r_inputs


def _construct_prompt(base_prompt: str, r_inputs: Dict[str, str]) -> str:
    """Construct final prompt with R inputs embedded."""
    # Simple concatenation - can be enhanced with template system
    prompt_parts = [base_prompt, "\n\n---\n\n"]
    
    for r_key, content in r_inputs.items():
        prompt_parts.append(f"## {r_key} Input\n\n{content}\n\n---\n\n")
    
    return "".join(prompt_parts)


def _check_and_enqueue_conversion_if_needed(
    db,
    run_id: str,
    doc_id: str,
    task_id: Optional[str],
    chain_version_id: Optional[str],
    completed_step_index: int,
    markdown_output_path: Path,
    storage_base: Path
):
    """
    Check if all steps are complete for this document, and if so,
    enqueue a conversion job if the workflow's export format requires it.
    """
    if not chain_version_id:
        logger.debug(f"No chain_version_id provided, skipping conversion check")
        return
        
    try:
        from external.ai_bulk_doc_analysis.models import ChainVersion as DBChainVersion, StepResult as DBStepResult
        from external.ai_bulk_doc_analysis.models import Run as DBRun, WorkflowVersion, ExportProfile
        
        # Get chain to know total step count
        chain_version = db.query(DBChainVersion).filter(
            DBChainVersion.chain_version_id == chain_version_id
        ).first()
        if not chain_version:
            logger.warning(f"Chain version {chain_version_id} not found")
            return
        
        total_steps = chain_version.step_count
        
        # Check if this was the last step
        if completed_step_index < total_steps:
            # Not the last step, nothing to do
            return
        
        # Get all step results for this document to verify all are complete
        query = db.query(DBStepResult).filter(
            DBStepResult.run_id == run_id,
            DBStepResult.doc_id == doc_id,
            DBStepResult.status == "SUCCESS"
        )
        if task_id:
            query = query.filter(DBStepResult.task_id == task_id)
        
        completed_steps = query.count()
        if completed_steps < total_steps:
            # Not all steps complete yet
            logger.debug(f"Not all steps complete yet for doc {doc_id}: {completed_steps}/{total_steps}")
            return
        
        # All steps complete! Check if conversion is needed
        # Get workflow version and export profile
        db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
        if not db_run or not db_run.workflow_version_id:
            logger.debug(f"No workflow_version_id for run {run_id}, skipping conversion")
            return
        
        workflow_version = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_version_id == db_run.workflow_version_id
        ).first()
        if not workflow_version or not workflow_version.export_profile_id:
            logger.debug(f"No export_profile_id for workflow version {db_run.workflow_version_id}")
            return
        
        export_profile = db.query(ExportProfile).filter(
            ExportProfile.export_profile_id == workflow_version.export_profile_id
        ).first()
        if not export_profile:
            logger.warning(f"Export profile {workflow_version.export_profile_id} not found")
            return
        
        export_format = export_profile.format
        
        # Only convert if format requires conversion (not MD or JSON)
        if export_format not in ['PDF', 'DOCX', 'CSV', 'XLSX']:
            logger.debug(f"Export format {export_format} doesn't require conversion")
            return
        
        # Enqueue conversion job
        try:
            import redis
            from rq import Queue
            
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            redis_client = redis.from_url(redis_url)
            conversion_queue = Queue("conversion", connection=redis_client)
            
            # Import the conversion worker wrapper function (from rq_worker, not the raw implementation)
            # Use string import path for RQ serialization
            from external.ai_bulk_doc_analysis.workers import rq_worker
            
            # Prepare job data
            job_data = {
                "run_id": run_id,
                "doc_id": doc_id,
                "export_format": export_format,
                "markdown_path": str(markdown_output_path.relative_to(storage_base)),
                "idempotency_key": f"convert_output:{run_id}:{doc_id}:{task_id or 'main'}",
            }
            
            job_id = f"convert_output_{run_id}_{doc_id}" + (f"_{task_id}" if task_id else "")
            conversion_queue.enqueue(
                rq_worker.convert_output_format_job,
                job_data,
                job_id=job_id
            )
            
            logger.info(f"Enqueued {export_format} conversion job for run {run_id}, doc {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to enqueue conversion job: {e}", exc_info=True)
            # Don't fail the step execution if conversion enqueue fails
            
    except Exception as e:
        logger.error(f"Error checking/enqueueing conversion: {e}", exc_info=True)
        # Don't fail the step execution if this check fails


def _update_run_status_if_complete(db, run_id: str, chain_version_id: Optional[str]):
    """
    Check if all steps for all documents/tasks in a run are complete,
    and update run status to COMPLETE if so.
    """
    if not chain_version_id:
        return
    
    try:
        from external.ai_bulk_doc_analysis.models import Run as DBRun, ChainVersion as DBChainVersion
        from external.ai_bulk_doc_analysis.models import StepResult as DBStepResult, ExecutionTask as DBExecutionTask
        from datetime import datetime
        
        db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
        if not db_run:
            return
        
        # Skip if already complete or error
        if db_run.status in ['COMPLETE', 'ERROR']:
            return
        
        # Get chain to know total step count
        chain_version = db.query(DBChainVersion).filter(
            DBChainVersion.chain_version_id == chain_version_id
        ).first()
        if not chain_version:
            return
        
        total_steps = chain_version.step_count
        
        # Check if CSV workflow (has execution_tasks)
        tasks = db.query(DBExecutionTask).filter(DBExecutionTask.run_id == run_id).all()
        is_csv_workflow = len(tasks) > 0
        
        if is_csv_workflow:
            # CSV workflow: check all tasks have all steps complete
            all_complete = True
            has_error = False
            
            for task in tasks:
                task_results = db.query(DBStepResult).filter(
                    DBStepResult.run_id == run_id,
                    DBStepResult.task_id == task.task_id
                ).all()
                
                completed_count = sum(1 for sr in task_results if sr.status == 'SUCCESS')
                error_count = sum(1 for sr in task_results if sr.status == 'ERROR')
                
                if error_count > 0:
                    has_error = True
                    all_complete = False
                    break
                
                if completed_count < total_steps:
                    all_complete = False
                    break
            
            if all_complete:
                db_run.status = "COMPLETE"
                db_run.completed_at = datetime.utcnow()
                db.commit()
                logger.info(f"Updated run {run_id} status to COMPLETE (CSV workflow)")
            elif has_error:
                db_run.status = "ERROR"
                db.commit()
                logger.info(f"Updated run {run_id} status to ERROR (CSV workflow)")
        else:
            # Non-CSV workflow: check all documents have all steps complete
            step_results = db.query(DBStepResult).filter(DBStepResult.run_id == run_id).all()
            
            if not step_results:
                return
            
            # Group by doc_id
            from collections import defaultdict
            doc_steps = defaultdict(list)
            for sr in step_results:
                doc_steps[sr.doc_id].append(sr)
            
            all_complete = True
            has_error = False
            
            for doc_id, results in doc_steps.items():
                completed_count = sum(1 for sr in results if sr.status == 'SUCCESS')
                error_count = sum(1 for sr in results if sr.status == 'ERROR')
                
                if error_count > 0:
                    has_error = True
                    all_complete = False
                    break
                
                if completed_count < total_steps:
                    all_complete = False
                    break
            
            if all_complete:
                db_run.status = "COMPLETE"
                db_run.completed_at = datetime.utcnow()
                db.commit()
                logger.info(f"Updated run {run_id} status to COMPLETE")
            elif has_error:
                db_run.status = "ERROR"
                db.commit()
                logger.info(f"Updated run {run_id} status to ERROR")
                
    except Exception as e:
        logger.error(f"Error updating run status: {e}", exc_info=True)
        # Don't fail the step execution if status update fails


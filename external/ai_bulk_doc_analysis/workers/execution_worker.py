"""
Execution Worker - Handles Claude API step execution jobs.
"""
import logging
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
    step_index = job_data["step_index"]
    task_id = job_data.get("task_id")  # NEW: For CSV workflows
    
    logger.info(f"Starting step {step_index} execution for run {run_id}, doc {doc_id}" + (f", task {task_id}" if task_id else ""))
    
    try:
        # Load required R inputs
        r_inputs = _load_r_inputs(
            storage_base=storage_base,
            run_id=run_id,
            doc_id=doc_id,
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


def _load_r_inputs(storage_base: Path, run_id: str, doc_id: str, required_inputs: list, task_id: Optional[str] = None) -> Dict[str, str]:
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
                # Non-CSV: Find R0 in session docs
                r0_path = storage_base / "sessions" / "docs" / doc_id / "R0.md"
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


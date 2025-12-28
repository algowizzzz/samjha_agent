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
    
    logger.info(f"Starting step {step_index} execution for run {run_id}, doc {doc_id}")
    
    try:
        # TODO: Check idempotency - if StepResult already SUCCESS, skip
        # existing = db.get_step_result(run_id, doc_id, step_index)
        # if existing and existing.status == "SUCCESS":
        #     logger.info(f"Step already completed, skipping")
        #     return {"status": "SKIPPED", "reason": "already_complete"}
        
        # TODO: Update StepResult status to RUNNING in DB
        # db.update_step_result_status(run_id, doc_id, step_index, "RUNNING")
        
        # Load required R inputs
        r_inputs = _load_r_inputs(
            storage_base=storage_base,
            run_id=run_id,
            doc_id=doc_id,
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
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    output_text += block_text
        
        output_text = response.text
        usage = response.usage
        
        # Extract token usage
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or 0 if usage else 0
        output_tokens = getattr(usage, "output_tokens", None) or 0 if usage else 0
        
        # Save R(n) output
        r_key = f"R{step_index}"
        run_dir = storage_base / "runs" / run_id / "docs" / doc_id
        run_dir.mkdir(parents=True, exist_ok=True)
        r_path = run_dir / f"{r_key}.md"
        r_path.write_text(output_text, encoding='utf-8')
        
        # TODO: Update StepResult in DB
        # db.update_step_result(
        #     run_id=run_id,
        #     doc_id=doc_id,
        #     step_index=step_index,
        #     status="SUCCESS",
        #     input_tokens=input_tokens,
        #     output_tokens=output_tokens,
        #     model=model,
        #     max_tokens=model_config.get("max_tokens"),
        #     temperature=model_config.get("temperature"),
        #     output_object_key=str(r_path)
        # )
        
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


def _load_r_inputs(storage_base: Path, run_id: str, doc_id: str, required_inputs: list) -> Dict[str, str]:
    """Load R0, R1, ... inputs from storage."""
    r_inputs = {}
    
    for r_key in required_inputs:
        # R0 comes from session docs, R1+ from run outputs
        if r_key == "R0":
            # Find R0 in session docs
            # TODO: Get session_id from run, then load R0
            # For now, assume path structure
            r0_path = storage_base / "sessions" / "docs" / doc_id / "R0.md"
        else:
            # R1, R2, ... from previous steps
            step_num = int(r_key[1:])  # Extract number from "R1"
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


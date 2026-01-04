"""
RQ worker functions for async job execution.
These are the entry points that RQ calls.
"""
import logging
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from external.ai_bulk_doc_analysis.db_service import BulkDocDBService
from external.ai_bulk_doc_analysis.workers.conversion_worker import convert_doc_job as convert_impl
from external.ai_bulk_doc_analysis.workers.execution_worker import execute_step_job as execute_impl
from external.ai_bulk_doc_analysis.workers.output_conversion_worker import convert_output_format_job as convert_output_impl

logger = logging.getLogger(__name__)


def convert_doc_job(job_data: Dict[str, Any]):
    """
    RQ job handler for CONVERT_DOC.
    
    Args:
        job_data: {
            "doc_id": str,
            "session_id": str,
            "object_storage_key": str,
            "idempotency_key": str
        }
    """
    try:
        # Get storage base path
        storage_base = Path(os.getenv("STORAGE_BASE", "data/ai_bulk_doc_analysis"))
        
        # Initialize DB service
        db_service = BulkDocDBService(storage_base=storage_base)
        
        # Update status to PROCESSING
        db_service.update_document_status(doc_id=job_data["doc_id"], status="PROCESSING")
        
        # Run conversion
        result = convert_impl(job_data, storage_base)
        
        # Update status to CONVERTED
        db_service.update_document_status(
            doc_id=job_data["doc_id"],
            status="CONVERTED",
            converted_md_path=result.get("r0_path"),
        )
        
        logger.info(f"Conversion job completed for doc {job_data['doc_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Conversion job failed for doc {job_data.get('doc_id')}: {e}", exc_info=True)
        # Update status to ERROR
        try:
            db_service = BulkDocDBService()
            db_service.update_document_status(
                doc_id=job_data["doc_id"],
                status="ERROR",
                error_code="CONVERSION_FAILED",
                error_message=str(e),
            )
        except:
            pass
        raise


def execute_step_job(job_data: Dict[str, Any]):
    """
    RQ job handler for EXECUTE_STEP.
    
    Args:
        job_data: {
            "run_id": str,
            "doc_id": str,
            "step_index": int,
            "chain_version_id": str,
            "required_inputs": List[str],
            "prompt": str,
            "model_config": Dict,
            "idempotency_key": str
        }
    """
    try:
        # Get storage base path
        storage_base = Path(os.getenv("STORAGE_BASE", "data/ai_bulk_doc_analysis"))
        
        # Initialize DB service
        db_service = BulkDocDBService(storage_base=storage_base)
        
        # Check idempotency - skip if already SUCCESS
        # (This is handled by unique constraint, but double-check here)
        # We'll update status in the worker implementation
        
        # Run execution
        result = execute_impl(job_data, storage_base)
        
        # Update step result in DB
        db_service.update_step_result(
            run_id=job_data["run_id"],
            doc_id=job_data["doc_id"],
            step_index=job_data["step_index"],
            status="SUCCESS",
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            model=job_data.get("model_config", {}).get("model", "claude-3-haiku-20240307"),
            max_tokens=job_data.get("model_config", {}).get("max_tokens"),
            temperature=job_data.get("model_config", {}).get("temperature"),
            latency_ms=result.get("latency_ms"),
            output_object_key=result.get("output_path"),
        )
        
        logger.info(f"Execution job completed for {job_data['run_id']}/{job_data['doc_id']}/step{job_data['step_index']}")
        return result
        
    except Exception as e:
        logger.error(f"Execution job failed: {e}", exc_info=True)
        # Update status to ERROR
        try:
            db_service = BulkDocDBService()
            db_service.update_step_result(
                run_id=job_data["run_id"],
                doc_id=job_data["doc_id"],
                step_index=job_data["step_index"],
                status="ERROR",
            error_code="EXECUTION_FAILED",
            error_message=str(e),
            )
        except:
            pass
        raise


def convert_output_format_job(job_data: Dict[str, Any]):
    """
    RQ job handler for converting markdown output to PDF/DOCX/CSV format.
    
    Args:
        job_data: {
            "run_id": str,
            "doc_id": str,
            "export_format": str,  # 'PDF', 'DOCX', 'CSV'
            "markdown_path": str,  # Relative path to the markdown file
            "idempotency_key": str
        }
    """
    try:
        # Get storage base path
        storage_base = Path(os.getenv("STORAGE_BASE", "data/ai_bulk_doc_analysis"))
        
        # Run conversion
        result = convert_output_impl(job_data, storage_base)
        
        logger.info(f"Output conversion job completed for run {job_data['run_id']}, doc {job_data['doc_id']} to {job_data['export_format']}")
        return result
        
    except Exception as e:
        logger.error(f"Output conversion job failed: {e}", exc_info=True)
        raise


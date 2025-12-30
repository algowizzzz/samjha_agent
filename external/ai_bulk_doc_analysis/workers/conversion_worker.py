"""
Conversion Worker - Handles file ingestion (programmatic and vision).
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def convert_doc_job(job_data: Dict[str, Any], storage_base: Path):
    """
    RQ job function for file ingestion.
    
    Args:
        job_data: {
            "doc_id": str,
            "session_id": str,
            "object_storage_key": str,  # Path to raw file
            "ingestion_profile_id": str,  # NEW: Ingestion profile ID
            "idempotency_key": str
        }
        storage_base: Base path for storage
    """
    doc_id = job_data["doc_id"]
    session_id = job_data["session_id"]
    object_storage_key = job_data["object_storage_key"]
    ingestion_profile_id = job_data.get("ingestion_profile_id")
    
    logger.info(f"Starting ingestion for doc {doc_id}")
    
    # Load file from storage
    file_path = storage_base / object_storage_key
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get ingestion profile
    from external.ai_bulk_doc_analysis.ingestion_service import IngestionService
    from external.ai_bulk_doc_analysis.db_service import init_db
    
    init_db()
    ingestion_service = IngestionService()
    
    if ingestion_profile_id:
        ingestion_profile = ingestion_service.get_ingestion_profile(ingestion_profile_id)
        if not ingestion_profile:
            raise ValueError(f"Ingestion profile {ingestion_profile_id} not found")
    else:
        # Fallback: create default profile for PDF (backward compatibility)
        from external.ai_bulk_doc_analysis.models import IngestionProfile
        from external.ai_bulk_doc_analysis.db_service import get_db_session
        with get_db_session() as db:
            # Try to find default PDF profile
            default_profile = db.query(IngestionProfile).filter(
                IngestionProfile.name == "Default PDF",
                IngestionProfile.mode == "programmatic"
            ).first()
            if not default_profile:
                # Create default profile
                ingestion_profile = ingestion_service.create_ingestion_profile(
                    name="Default PDF",
                    accepted_input_types=["PDF"],
                    mode="programmatic"
                )
            else:
                ingestion_profile = default_profile
    
    # Detect file type
    file_type = ingestion_service._detect_file_type(file_path)
    
    # Special handling for CSV
    if file_type == 'CSV':
        # CSV creates execution_tasks, not R0.md
        # This will be handled in run creation
        from external.ai_bulk_doc_analysis.db_service import get_db_session
        from external.ai_bulk_doc_analysis.models import Document
        
        with get_db_session() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                doc.status = "CONVERTED"  # CSV is "converted" (tasks will be created)
                db.commit()
        
        logger.info(f"CSV file {doc_id} marked as CONVERTED (tasks will be created during run)")
        return {"status": "CSV_PROCESSED", "file_type": "CSV"}
    
    # Ingest file
    r0_content, metadata = ingestion_service.ingest_file(file_path, ingestion_profile)
    
    # Save R0.md artifact
    output_dir = storage_base / "sessions" / session_id / "docs" / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    r0_path = output_dir / "R0.md"
    r0_path.write_text(r0_content, encoding='utf-8')
    
    # Update document status in database
    try:
        from external.ai_bulk_doc_analysis.db_service import get_db_session
        from external.ai_bulk_doc_analysis.models import Document
        
        with get_db_session() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                doc.status = "CONVERTED"
                doc.converted_md_path = str(r0_path)
                db.commit()
                logger.info(f"Updated document {doc_id} status to CONVERTED")
    except Exception as e:
        logger.warning(f"Failed to update document status: {e}")
    
    logger.info(f"Ingestion complete for doc {doc_id}, method: {metadata.get('method', 'unknown')}")
    return {"status": "SUCCESS", "r0_path": str(r0_path), "metadata": metadata}


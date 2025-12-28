"""
Conversion Worker - Handles PDF → Markdown conversion jobs.
"""
import logging
from pathlib import Path
from typing import Dict, Any

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logger = logging.getLogger(__name__)


def convert_doc_job(job_data: Dict[str, Any]):
    """
    RQ job function for PDF → Markdown conversion.
    
    Args:
        job_data: {
            "doc_id": str,
            "session_id": str,
            "object_storage_key": str,  # Path to raw PDF
            "idempotency_key": str
        }
    """
    from pathlib import Path
    
    # Get storage base from config
    storage_base = Path("data/ai_bulk_doc_analysis")
    doc_id = job_data["doc_id"]
    session_id = job_data["session_id"]
    object_storage_key = job_data["object_storage_key"]
    
    logger.info(f"Starting conversion for doc {doc_id}")
    
    # Load PDF from storage
    pdf_path = storage_base / object_storage_key
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Convert PDF to Markdown
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber not installed")
    
    markdown_content = _convert_pdf_to_markdown(pdf_path)
    
    # Save R0.md artifact
    output_dir = storage_base / "sessions" / session_id / "docs" / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    r0_path = output_dir / "R0.md"
    r0_path.write_text(markdown_content, encoding='utf-8')
    
    # Update document status in database
    try:
        from external.ai_bulk_doc_analysis.db_service import get_db_session, init_db
        from external.ai_bulk_doc_analysis.models import Document
        
        init_db()
        with get_db_session() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                doc.status = "CONVERTED"
                doc.converted_md_path = str(r0_path)
                db.commit()
                logger.info(f"Updated document {doc_id} status to CONVERTED")
    except Exception as e:
        logger.warning(f"Failed to update document status: {e}")
    
    logger.info(f"Conversion complete for doc {doc_id}")
    return {"status": "SUCCESS", "r0_path": str(r0_path)}


def _convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert PDF to Markdown using pdfplumber."""
    markdown_parts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                markdown_parts.append(f"## Page {page_num}\n\n{text}\n")
    
    return "\n".join(markdown_parts)


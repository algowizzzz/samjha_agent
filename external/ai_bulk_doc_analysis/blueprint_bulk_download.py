"""
Bulk Download Endpoint - Feature #3
Adds download-all functionality for runs.
"""
import zipfile
import io
from flask import Blueprint, jsonify, send_file
from pathlib import Path
from typing import Optional

# This will be integrated into blueprint.py


def add_bulk_download_route(bp: Blueprint, get_service):
    """
    Add bulk download route to blueprint.
    
    Usage in blueprint.py:
        from external.ai_bulk_doc_analysis.blueprint_bulk_download import add_bulk_download_route
        add_bulk_download_route(bp, get_service)
    """
    
    @bp.route("/api/bulk-doc-analysis/runs/<run_id>/download-all", methods=["GET"])
    def api_download_all_outputs(run_id: str):
        """Download all document outputs for a run as a ZIP archive."""
        from flask import session
        from external.ai_bulk_doc_analysis.blueprint import _current_user_session
        
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            svc = get_service()
            
            # Get run
            run = svc.runs.get(run_id)
            if not run:
                return jsonify({"error": "Run not found"}), 404
            
            # Get chain to determine final step
            chain = svc.get_chain(run.chain_version_id)
            if not chain:
                return jsonify({"error": "Chain not found"}), 404
            
            # Get all documents in the run
            docs = [d for d in svc.documents.values() if d.session_id == run.session_id]
            
            if not docs:
                return jsonify({"error": "No documents found for this run"}), 404
            
            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for doc in docs:
                    # Get final output path
                    output_path = svc.get_final_output_path(run_id, doc.doc_id, chain)
                    
                    if output_path and output_path.exists():
                        # Add to ZIP with readable filename
                        base_name = Path(doc.original_filename).stem
                        zip_filename = f"{base_name}_output.md"
                        zip_file.write(str(output_path), zip_filename)
            
            zip_buffer.seek(0)
            
            # Return ZIP file
            return send_file(
                zip_buffer,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"run_{run_id}_outputs.zip"
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Bulk download error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500


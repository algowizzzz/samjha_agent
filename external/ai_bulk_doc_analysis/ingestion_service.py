"""
Ingestion Service - Handles file ingestion (programmatic and vision).
"""
import logging
import base64
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logging.warning("Pillow not available")

try:
    from external.platform.llm.client import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM client not available")

from .models import IngestionProfile
from .db_service import get_db_session

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for file ingestion (programmatic and vision)."""
    
    def create_ingestion_profile(
        self,
        name: str,
        accepted_input_types: List[str],
        mode: str,
        vision_prompt: Optional[str] = None
    ) -> IngestionProfile:
        """
        Create an ingestion profile.
        
        Args:
            name: Profile name
            accepted_input_types: List of accepted file types ['PDF', 'DOCX', 'TXT', 'MD', 'CSV']
            mode: 'programmatic' or 'vision'
            vision_prompt: Vision prompt text (required if mode='vision')
            
        Returns:
            Created IngestionProfile object
            
        Raises:
            ValueError: If validation fails
        """
        # Validate mode
        if mode not in ['programmatic', 'vision']:
            raise ValueError("Mode must be 'programmatic' or 'vision'")
        
        # Validate vision prompt
        if mode == 'vision' and not vision_prompt:
            raise ValueError("vision_prompt is required when mode='vision'")
        
        # Validate accepted types
        valid_types = ['PDF', 'DOCX', 'TXT', 'MD', 'CSV']
        for file_type in accepted_input_types:
            if file_type not in valid_types:
                raise ValueError(f"Invalid file type: {file_type}. Valid types: {valid_types}")
        
        with get_db_session() as db:
            ingestion_profile_id = f"ing_{uuid.uuid4().hex[:12]}"
            profile = IngestionProfile(
                ingestion_profile_id=ingestion_profile_id,
                name=name,
                accepted_input_types=accepted_input_types,
                mode=mode,
                vision_prompt=vision_prompt
            )
            db.add(profile)
            db.commit()
            
            logger.info(f"Created ingestion profile {ingestion_profile_id}")
            # Return a simple object with the data (avoiding session binding issues)
            class ProfileResult:
                pass
            result = ProfileResult()
            result.ingestion_profile_id = ingestion_profile_id
            result.name = name
            result.accepted_input_types = accepted_input_types
            result.mode = mode
            result.vision_prompt = vision_prompt
            return result
    
    def get_ingestion_profile(self, ingestion_profile_id: str) -> Optional[IngestionProfile]:
        """Get ingestion profile by ID."""
        with get_db_session() as db:
            return db.query(IngestionProfile).filter(
                IngestionProfile.ingestion_profile_id == ingestion_profile_id
            ).first()
    
    def list_ingestion_profiles(self) -> List[IngestionProfile]:
        """List all ingestion profiles."""
        with get_db_session() as db:
            return db.query(IngestionProfile).all()
    
    def ingest_file(
        self,
        file_path: Path,
        ingestion_profile: IngestionProfile
    ) -> Tuple[str, Dict]:
        """
        Ingest a file and return R0 content.
        
        Args:
            file_path: Path to file
            ingestion_profile: IngestionProfile object
            
        Returns:
            Tuple of (r0_content, metadata_dict)
            
        Raises:
            ValueError: If file type not supported or mode invalid
            RuntimeError: If ingestion fails
        """
        file_type = self._detect_file_type(file_path)
        
        # Validate file type
        if file_type not in ingestion_profile.accepted_input_types:
            raise ValueError(f"File type {file_type} not in accepted types: {ingestion_profile.accepted_input_types}")
        
        # Route to appropriate ingestion method
        if ingestion_profile.mode == 'programmatic':
            return self._ingest_programmatic(file_path, file_type)
        elif ingestion_profile.mode == 'vision':
            if file_type != 'PDF':
                raise ValueError("Vision ingestion only supports PDF files")
            return self._ingest_vision(file_path, ingestion_profile.vision_prompt)
        else:
            raise ValueError(f"Unknown ingestion mode: {ingestion_profile.mode}")
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension."""
        ext = file_path.suffix.lower()
        mapping = {
            '.pdf': 'PDF',
            '.docx': 'DOCX',
            '.txt': 'TXT',
            '.md': 'MD',
            '.csv': 'CSV'
        }
        return mapping.get(ext, 'UNKNOWN')
    
    def _ingest_programmatic(self, file_path: Path, file_type: str) -> Tuple[str, Dict]:
        """
        Programmatic ingestion - extract text using libraries.
        
        Returns:
            Tuple of (r0_content, metadata)
        """
        metadata = {"method": "programmatic", "file_type": file_type}
        
        if file_type == 'PDF':
            if not PDFPLUMBER_AVAILABLE:
                raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")
            
            markdown_parts = []
            markdown_parts.append(f"# Document: {file_path.name}\n\n")
            
            # Validate file is actually a PDF by checking magic bytes
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        raise RuntimeError(f"File '{file_path.name}' does not appear to be a valid PDF file. PDF files must start with '%PDF' magic bytes. This file appears to be a different format.")
            except Exception as e:
                if "magic bytes" in str(e) or "does not appear" in str(e):
                    raise
                # If it's a different error, let pdfplumber handle it
                pass
            
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        markdown_parts.append(f"## Page {page_num}\n\n")
                        
                        # Extract text
                        text = page.extract_text()
                        if text:
                            text = text.strip()
                            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                            markdown_parts.append('\n\n'.join(paragraphs))
                            markdown_parts.append('\n\n')
                        
                        # Extract tables
                        tables = page.extract_tables()
                        if tables:
                            for table_num, table in enumerate(tables, start=1):
                                markdown_parts.append(f"### Table {table_num} (Page {page_num})\n\n")
                                if table and len(table) > 0:
                                    if table[0]:
                                        header = "| " + " | ".join(str(cell or "") for cell in table[0]) + " |"
                                        separator = "| " + " | ".join("---" for _ in table[0]) + " |"
                                        markdown_parts.append(header)
                                        markdown_parts.append(separator)
                                    
                                    for row in table[1:]:
                                        if row:
                                            row_md = "| " + " | ".join(str(cell or "") for cell in row) + " |"
                                            markdown_parts.append(row_md)
                                    markdown_parts.append('\n\n')
                
                r0_content = ''.join(markdown_parts)
                metadata["page_count"] = len(pdf.pages) if 'pdf' in locals() else 0
                return r0_content, metadata
                
            except RuntimeError:
                # Re-raise validation errors as-is
                raise
            except Exception as e:
                logger.error(f"Error converting PDF {file_path}: {e}", exc_info=True)
                # Provide more helpful error message
                error_msg = str(e)
                if "No /Root object" in error_msg or "really a PDF" in error_msg:
                    raise RuntimeError(f"PDF conversion error: File '{file_path.name}' is not a valid PDF file. Please ensure the file is a properly formatted PDF document.")
                raise RuntimeError(f"PDF conversion error: {error_msg}")
        
        elif file_type == 'DOCX':
            if not DOCX_AVAILABLE:
                raise RuntimeError("python-docx not installed. Install with: pip install python-docx")
            
            markdown_parts = []
            markdown_parts.append(f"# Document: {file_path.name}\n\n")
            
            try:
                doc = DocxDocument(file_path)
                
                for para in doc.paragraphs:
                    if para.text.strip():
                        # Simple heading detection (if style indicates heading)
                        if para.style.name.startswith('Heading'):
                            level = para.style.name.replace('Heading ', '')
                            markdown_parts.append(f"{'#' * int(level)} {para.text}\n\n")
                        else:
                            markdown_parts.append(f"{para.text}\n\n")
                
                # Extract tables
                for table_num, table in enumerate(doc.tables, start=1):
                    markdown_parts.append(f"### Table {table_num}\n\n")
                    if table.rows:
                        # Header row
                        header_row = table.rows[0]
                        header = "| " + " | ".join(str(cell.text or "") for cell in header_row.cells) + " |"
                        separator = "| " + " | ".join("---" for _ in header_row.cells) + " |"
                        markdown_parts.append(header)
                        markdown_parts.append(separator)
                        
                        # Data rows
                        for row in table.rows[1:]:
                            row_md = "| " + " | ".join(str(cell.text or "") for cell in row.cells) + " |"
                            markdown_parts.append(row_md)
                        markdown_parts.append('\n\n')
                
                r0_content = ''.join(markdown_parts)
                metadata["paragraph_count"] = len(doc.paragraphs)
                return r0_content, metadata
                
            except Exception as e:
                logger.error(f"Error converting DOCX {file_path}: {e}", exc_info=True)
                raise RuntimeError(f"DOCX conversion error: {str(e)}")
        
        elif file_type in ['TXT', 'MD']:
            # Direct read
            try:
                r0_content = file_path.read_text(encoding='utf-8')
                metadata["char_count"] = len(r0_content)
                return r0_content, metadata
            except Exception as e:
                logger.error(f"Error reading {file_type} {file_path}: {e}", exc_info=True)
                raise RuntimeError(f"File read error: {str(e)}")
        
        elif file_type == 'CSV':
            # CSV is handled specially - returns metadata only, actual ingestion creates tasks
            # This method is called but CSV should be handled in conversion worker
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                metadata["row_count"] = len(df)
                metadata["column_count"] = len(df.columns)
                metadata["columns"] = df.columns.tolist()
                # Return empty content - CSV ingestion creates tasks, not R0
                return "", metadata
            except Exception as e:
                logger.error(f"Error reading CSV {file_path}: {e}", exc_info=True)
                raise RuntimeError(f"CSV read error: {str(e)}")
        
        else:
            raise ValueError(f"Unsupported file type for programmatic ingestion: {file_type}")
    
    def _ingest_vision(self, file_path: Path, vision_prompt: str) -> Tuple[str, Dict]:
        """
        Vision ingestion - convert PDF to images and call Claude Vision API.
        
        Args:
            file_path: Path to PDF file
            vision_prompt: Vision prompt text
            
        Returns:
            Tuple of (r0_content, metadata)
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image not installed. Install with: pip install pdf2image")
        
        if not PILLOW_AVAILABLE:
            raise RuntimeError("Pillow not installed. Install with: pip install Pillow")
        
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM client not available")
        
        metadata = {"method": "vision", "file_type": "PDF"}
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(str(file_path))
            metadata["page_count"] = len(images)
            
            # Prepare messages for Claude Vision API
            messages = []
            
            # Add images (base64 encoded)
            for img in images:
                # Convert PIL Image to base64
                import io
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                messages.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })
            
            # Add text prompt
            messages.append({
                "type": "text",
                "text": vision_prompt
            })
            
            # Call Claude Vision API
            llm_client = get_llm_client()
            if not llm_client.is_available():
                raise RuntimeError("LLM client not initialized. Check ANTHROPIC_API_KEY.")
            
            # Use vision-capable model
            response = llm_client.client.messages.create(
                model="claude-3-opus-20240229",  # Vision-capable model
                messages=[{"role": "user", "content": messages}],
                max_tokens=4096
            )
            
            # Extract text from response
            r0_content = ""
            for block in getattr(response, "content", []) or []:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str):
                        r0_content += block_text
            
            # Get token usage
            usage = getattr(response, "usage", None)
            if usage:
                metadata["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
                metadata["output_tokens"] = getattr(usage, "output_tokens", 0) or 0
            
            logger.info(f"Vision ingestion completed for {file_path.name}, {metadata.get('page_count', 0)} pages")
            return r0_content, metadata
            
        except Exception as e:
            logger.error(f"Vision ingestion failed for {file_path}: {e}", exc_info=True)
            raise RuntimeError(f"Vision ingestion error: {str(e)}")
    
    def estimate_ingestion_tokens(
        self,
        file_path: Path,
        ingestion_profile: IngestionProfile
    ) -> int:
        """
        Estimate token count for ingestion.
        
        Args:
            file_path: Path to file
            ingestion_profile: IngestionProfile object
            
        Returns:
            Estimated token count
        """
        if ingestion_profile.mode == 'programmatic':
            # For programmatic: estimate based on file size
            # Rough estimate: 1 token ≈ 4 characters
            file_size = file_path.stat().st_size
            estimated_chars = file_size  # Conservative estimate
            return estimated_chars // 4
        
        elif ingestion_profile.mode == 'vision':
            # For vision: prompt tokens + image payload estimate
            # Rough estimate: ~1000 tokens per image (adjust based on size)
            try:
                if PDF2IMAGE_AVAILABLE:
                    images = convert_from_path(str(file_path), first_page=1, last_page=1)
                    image_count = len(images) if images else 1
                else:
                    # Fallback: estimate 1 page
                    image_count = 1
                
                # Estimate prompt tokens (rough: 1 token ≈ 4 chars)
                prompt_tokens = len(ingestion_profile.vision_prompt or "") // 4
                # Estimate image tokens (~1000 per image, adjust based on size)
                image_tokens = image_count * 1000
                return prompt_tokens + image_tokens
            except Exception:
                # Fallback estimate
                return 5000


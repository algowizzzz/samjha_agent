from __future__ import annotations

import mimetypes
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:  # Optional dependency
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    pdfplumber = None

from tools.base_mcp_tool import BaseMCPTool

from .utils import ensure_directory, resolve_path


_IMAGE_OUTPUT_ROOT = Path("external/data/doc_review/images")
_DOCX_MEDIA_PREFIX = "word/media/"
_PPTX_MEDIA_PREFIX = "ppt/media/"


class ExtractImagesTool(BaseMCPTool):
    """Extract embedded images from a source document."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "extract_images",
            "description": "Extracts images from the source document and stores them under /images/.",
            "inputSchema": self.get_input_schema(),
            "outputSchema": self.get_output_schema(),
        }
        if config:
            tool_config.update(config)
        super().__init__(tool_config)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "source_path": {"type": "string"},
                "file_type": {
                    "type": "string",
                    "description": "Optional explicit file type (docx|pdf|pptx|md|...). Falls back to file extension.",
                },
            },
            "required": ["file_id", "source_path"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "images": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string"},
                            "path": {"type": "string"},
                            "page_number": {"type": ["integer", "null"]},
                            "caption": {"type": ["string", "null"]},
                            "mime_type": {"type": ["string", "null"]},
                        },
                        "required": ["image_id", "path"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["file_id", "images"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_id = arguments["file_id"]
        source_path = resolve_path(arguments["source_path"])
        file_type = (arguments.get("file_type") or source_path.suffix.lstrip(".")).lower()

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        images_dir = ensure_directory(_IMAGE_OUTPUT_ROOT / file_id)
        images: List[Dict[str, Any]] = []
        notes: str

        if file_type == "docx":
            images, notes = self._extract_from_package(source_path, images_dir, _DOCX_MEDIA_PREFIX, "docx")
        elif file_type == "pptx":
            images, notes = self._extract_from_package(source_path, images_dir, _PPTX_MEDIA_PREFIX, "pptx")
        elif file_type == "pdf":
            images, notes = self._extract_from_pdf(source_path, images_dir)
        else:
            images = []
            notes = f"No image extractor configured for {file_type or 'unknown'}."

        self.logger.info(
            "extract_images: file_id=%s path=%s images_found=%d",
            file_id,
            source_path,
            len(images),
        )

        return {
            "file_id": file_id,
            "images": images,
            "notes": notes,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _extract_from_package(
        self,
        source_path: Path,
        images_dir: Path,
        media_prefix: str,
        package_type: str,
    ) -> tuple[List[Dict[str, Any]], str]:
        extracted: List[Dict[str, Any]] = []

        with zipfile.ZipFile(source_path) as archive:
            for name in archive.namelist():
                if not name.startswith(media_prefix):
                    continue
                data = archive.read(name)
                ext = Path(name).suffix or ".bin"
                mime_guess, _ = mimetypes.guess_type(name)
                image_id = f"{source_path.stem}_img_{uuid4().hex[:8]}"
                target_path = images_dir / f"{image_id}{ext}"
                target_path.write_bytes(data)
                extracted.append(
                    {
                        "image_id": image_id,
                        "path": str(target_path),
                        "page_number": None,
                        "caption": None,
                        "mime_type": mime_guess,
                    }
                )

        note = f"Extracted {len(extracted)} images from {package_type.upper()} package."
        if not extracted:
            note = f"No embedded images found in {package_type.upper()} file."
        return extracted, note

    def _extract_from_pdf(self, source_path: Path, images_dir: Path) -> tuple[List[Dict[str, Any]], str]:
        if pdfplumber is None:  # type: ignore[name-defined]  # pragma: no cover - handled at runtime
            return [], "pdfplumber not installed; cannot extract images from PDF."

        extracted: List[Dict[str, Any]] = []
        with pdfplumber.open(str(source_path)) as pdf:  # type: ignore[name-defined]
            for page_index, page in enumerate(pdf.pages, start=1):
                for image_index, image in enumerate(page.images or [], start=1):
                    try:
                        obj = pdf.writer._get_xobject(image["name"])  # type: ignore[attr-defined]
                        image_data = obj["image"]
                        ext = obj["ext"]
                    except Exception:  # pragma: no cover - best effort extraction
                        continue

                    image_id = f"{source_path.stem}_p{page_index}_{uuid4().hex[:8]}"
                    target_path = images_dir / f"{image_id}.{ext}"
                    target_path.write_bytes(image_data)
                    extracted.append(
                        {
                            "image_id": image_id,
                            "path": str(target_path),
                            "page_number": page_index,
                            "caption": None,
                            "mime_type": mimetypes.types_map.get(f".{ext}"),
                        }
                    )

        note = f"Extracted {len(extracted)} images from PDF."
        if not extracted:
            note = "No extractable images found in PDF."
        return extracted, note

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

from .utils import build_heading_anchor, normalize_heading_level, slugify


class BuildIndexTool(BaseMCPTool):
    """Create the document index used across the workflow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "build_index",
            "description": "Builds the master index structure tying chunks to headings and metadata.",
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
                "file_metadata": {"type": "object"},
                "chunks": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "heading_structure": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "chunking_decision": {"type": "object"},
            },
            "required": ["file_metadata", "chunks"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {"type": "object"},
            },
            "required": ["index"],
            "additionalProperties": False,
        }

    def _build_sections_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build sections from chunks (when chunking is enabled)."""
        sections: List[Dict[str, Any]] = []
        for chunk in chunks:
            heading_path = chunk.get("heading_path") or [chunk.get("heading_text_original", "Section")]
            file_section_id = f"sec_{slugify('-'.join(heading_path))}_{chunk['order']:04d}"
            sections.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "file_section_id": file_section_id,
                    "heading_level": chunk.get("heading_level", "h2"),
                    "heading_text_original": chunk.get("heading_text_original", ""),
                    "heading_text_improved": None,
                    "heading_path": heading_path,
                    "anchor": build_heading_anchor(heading_path),
                    "template_mapping": {
                        "mapped": False,
                        "template_section_id": None,
                        "template_path": None,
                        "confidence": None,
                        "relevancy_score": None,
                    },
                    "relevancy_score_if_unmapped": None,
                    "status": "unmapped",
                    "order": chunk["order"],
                }
            )
        return sections

    def _build_sections_from_headings(
        self,
        heading_structure: List[Dict[str, Any]],
        target_level: str,
        single_chunk_id: str,
    ) -> List[Dict[str, Any]]:
        """Build sections from headings at target level (when chunking is disabled)."""
        sections: List[Dict[str, Any]] = []
        target_level_normalized = normalize_heading_level(target_level)

        # Filter headings at the target level
        target_headings = [
            h for h in heading_structure
            if h.get("heading_level", "").lower() == target_level_normalized
        ]

        # If no headings at target level, use h1 or first available heading
        if not target_headings:
            if heading_structure:
                target_headings = [heading_structure[0]]
                target_level_normalized = heading_structure[0].get("heading_level", "h1").lower()
            else:
                # Fallback: create a single section
                sections.append(
                    {
                        "chunk_id": single_chunk_id,
                        "file_section_id": "sec_document_0000",
                        "heading_level": "h1",
                        "heading_text_original": "Document",
                        "heading_text_improved": None,
                        "heading_path": ["Document"],
                        "anchor": build_heading_anchor(["Document"]),
                        "template_mapping": {
                            "mapped": False,
                            "template_section_id": None,
                            "template_path": None,
                            "confidence": None,
                            "relevancy_score": None,
                        },
                        "relevancy_score_if_unmapped": None,
                        "status": "unmapped",
                        "order": 0,
                    }
                )
                return sections

        # Build sections from target headings
        for idx, heading in enumerate(target_headings):
            heading_path = heading.get("heading_path", [heading.get("heading_text", "Section")])
            file_section_id = f"sec_{slugify('-'.join(heading_path))}_{idx:04d}"
            sections.append(
                {
                    "chunk_id": single_chunk_id,  # All sections link to the single chunk
                    "file_section_id": file_section_id,
                    "heading_level": heading.get("heading_level", target_level_normalized),
                    "heading_text_original": heading.get("heading_text", ""),
                    "heading_text_improved": None,
                    "heading_path": heading_path,
                    "anchor": build_heading_anchor(heading_path),
                    "template_mapping": {
                        "mapped": False,
                        "template_section_id": None,
                        "template_path": None,
                        "confidence": None,
                        "relevancy_score": None,
                    },
                    "relevancy_score_if_unmapped": None,
                    "status": "unmapped",
                    "order": idx,
                }
            )
        return sections

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        metadata = arguments["file_metadata"]
        chunks = arguments["chunks"]
        heading_structure = arguments.get("heading_structure", [])
        chunking_decision = arguments.get("chunking_decision", {})

        # Determine if chunking is disabled
        should_chunk = chunking_decision.get("should_chunk", True)
        strategy = chunking_decision.get("strategy", "").lower()
        context_aware_level = chunking_decision.get("context_aware_level", "h2")

        # If chunking is disabled, build sections from headings instead of chunks
        if not should_chunk or strategy == "none":
            if not chunks:
                self.logger.warning("build_index: no chunks available but chunking disabled")
                sections = []
            else:
                single_chunk_id = chunks[0]["chunk_id"]
                sections = self._build_sections_from_headings(
                    heading_structure, context_aware_level, single_chunk_id
                )
                self.logger.info(
                    "build_index: chunking disabled, built %d sections from headings at level %s",
                    len(sections),
                    context_aware_level,
                )
        else:
            # Normal path: build sections from chunks
            sections = self._build_sections_from_chunks(chunks)
            self.logger.info(
                "build_index: chunking enabled, built %d sections from %d chunks",
                len(sections),
                len(chunks),
            )

        stats = {
            "total_sections": len(sections),
            "mapped_sections": 0,
            "unmapped_sections": len(sections),
            "version": "v1",
        }

        index = {
            "file_id": metadata.get("file_id"),
            "md_file_id": metadata.get("md_file_id"),
            "template_id": None,
            "chunking": metadata.get("chunking", {}),
            "sections": sections,
            "toc": [],
            "stats": stats,
        }

        self.logger.info(
            "build_index: file_id=%s sections=%d",
            index["file_id"],
            len(sections),
        )

        return {"index": index}

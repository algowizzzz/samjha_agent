"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Assemble Documentation
"""

from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocAssembleDocTool(BaseMCPTool):
    """Assemble final markdown documentation from template and section drafts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_assemble_doc",
            "description": "Combine section drafts into a complete markdown document based on template structure.",
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
                "template": {
                    "type": "object",
                    "description": "Template structure from load_documentation_template.",
                },
                "section_drafts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_id": {"type": "string"},
                            "template_section_id": {"type": ["string", "null"]},
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                            "order": {"type": "integer"},
                        },
                    },
                    "description": "List of section drafts with content.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to include in document header.",
                },
            },
            "required": ["template", "section_drafts"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "markdown": {"type": "string", "description": "Complete markdown document."},
                "section_count": {"type": "integer"},
            },
            "required": ["markdown", "section_count"],
            "additionalProperties": False,
        }

    def _build_markdown(
        self, template: Dict[str, Any], section_drafts: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build markdown from template and sections."""
        lines = []
        
        # Add title/header
        title = metadata.get("title", "Model Documentation") if metadata else "Model Documentation"
        lines.append(f"# {title}\n")
        
        # Add metadata if provided
        if metadata:
            lines.append("---\n")
            for key, value in metadata.items():
                if key != "title" and value:
                    lines.append(f"**{key.title()}**: {value}\n")
            lines.append("---\n\n")
        
        # Build table of contents if template has sections
        template_sections = template.get("sections", [])
        if template_sections:
            lines.append("## Table of Contents\n\n")
            for section in template_sections:
                section_id = section.get("id", "")
                heading = section.get("title", section_id)
                lines.append(f"- [{heading}](#{section_id})\n")
            lines.append("\n")
        
        # Sort section drafts by order
        sorted_drafts = sorted(section_drafts, key=lambda x: x.get("order", 999))
        
        # Build sections
        section_map = {draft["section_id"]: draft for draft in sorted_drafts}
        
        for draft in sorted_drafts:
            heading = draft.get("heading", draft.get("section_id", "Section"))
            content = draft.get("content", "")
            order = draft.get("order", 999)
            
            # Use template section title if available
            template_section_id = draft.get("template_section_id")
            if template_section_id:
                template_section = next(
                    (s for s in template_sections if s.get("id") == template_section_id), None
                )
                if template_section:
                    heading = template_section.get("title", heading)
            
            # Add section
            lines.append(f"## {heading}\n\n")
            if content:
                lines.append(f"{content}\n\n")
            else:
                lines.append("*No content available for this section.*\n\n")
        
        return "".join(lines)

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        template = arguments["template"]
        section_drafts = arguments["section_drafts"]
        metadata = arguments.get("metadata")

        markdown = self._build_markdown(template, section_drafts, metadata)

        self.logger.info(
            f"model_doc_assemble_doc: sections={len(section_drafts)} output_length={len(markdown)}"
        )

        return {
            "markdown": markdown,
            "section_count": len(section_drafts),
        }


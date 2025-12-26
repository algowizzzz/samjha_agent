"""
Search Glossary Tool - Search domain glossary for term definitions.
"""

import re
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool


class SearchGlossaryTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "search_glossary",
            "description": "Search domain glossary for term definitions",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.domain_dir = Path(self.config.get("domain_directory", "domain_instructions"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["term", "domain"],
            "properties": {
                "term": {"type": "string"},
                "domain": {"type": "string"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["hits"],
            "properties": {
                "hits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["title", "snippet", "source_ref"],
                        "properties": {
                            "title": {"type": "string"},
                            "snippet": {"type": "string"},
                            "source_ref": {"type": "string"}
                        }
                    }
                }
            }
        }

    def execute(self, arguments):
        term = arguments["term"].lower()
        domain = arguments["domain"]
        
        # Find domain file
        domain_file = self.domain_dir / f"{domain}_domain.md"
        if not domain_file.exists():
            return {"hits": []}
        
        content = domain_file.read_text()
        hits = []
        
        # Parse metrics section
        metrics_match = re.search(r"## 5\) Metric dictionary.*?(?=## \d|\Z)", content, re.DOTALL)
        if metrics_match:
            metrics_text = metrics_match.group(0)
            # Find term in metrics
            for match in re.finditer(rf"- metric_name:\s*(\S+).*?(?=- metric_name:|\Z)", metrics_text, re.DOTALL):
                metric_block = match.group(0)
                metric_name = match.group(1)
                if term in metric_name.lower() or term in metric_block.lower():
                    definition_match = re.search(r"definition:\s*(.+?)(?=\n\s+-|\Z)", metric_block)
                    definition = definition_match.group(1).strip() if definition_match else "No definition"
                    hits.append({
                        "title": metric_name,
                        "snippet": definition,
                        "source_ref": f"{domain}_domain.md#metrics"
                    })
        
        # Parse entities section
        entities_match = re.search(r"## 4\) Core entities.*?(?=## \d|\Z)", content, re.DOTALL)
        if entities_match:
            entities_text = entities_match.group(0)
            if term in entities_text.lower():
                for match in re.finditer(r"- name:\s*(\S+).*?typical_grain:\s*(.+?)(?=\n\s+-|\Z)", entities_text, re.DOTALL):
                    entity_name = match.group(1)
                    grain = match.group(2)
                    if term in entity_name.lower():
                        hits.append({
                            "title": entity_name,
                            "snippet": f"Grain: {grain.strip()}",
                            "source_ref": f"{domain}_domain.md#entities"
                        })
        
        return {"hits": hits}


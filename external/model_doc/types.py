from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class CodeStructure(TypedDict, total=False):
    """AST structure extracted from a Python file"""
    file_path: str
    classes: List[Dict[str, Any]]  # name, methods, docstring, line_number
    functions: List[Dict[str, Any]]  # name, args, docstring, line_number
    imports: List[Dict[str, Any]]  # module, names, type (import/from)
    docstring: Optional[str]
    ast_errors: Optional[List[str]]


class FileMetadata(TypedDict, total=False):
    """Metadata for a single source file"""
    file_path: str
    file_size: int
    line_count: int
    encoding: Optional[str]
    ast_structure: Optional[CodeStructure]
    summary: Optional[str]  # LLM-generated summary


class CodebaseMetadata(TypedDict, total=False):
    """Overall codebase metadata"""
    codebase_path: str
    codebase_id: str
    file_count: int
    total_lines: int
    total_classes: int
    total_functions: int
    directory_structure: Dict[str, Any]  # Tree structure
    file_list: List[FileMetadata]


class HierarchicalSummary(TypedDict, total=False):
    """Hierarchical summary of the codebase"""
    module_hierarchy: Dict[str, Any]  # Tree structure with summaries
    component_summary: str  # Overall summary
    key_modules: List[str]
    key_classes: List[str]
    dependencies: List[str]


class SectionMapping(TypedDict, total=False):
    """Mapping of codebase sections to documentation template"""
    section_id: str
    template_section_id: Optional[str]
    template_path: Optional[List[str]]
    mapped: bool
    confidence: Optional[float]
    codebase_sections: List[str]  # File/module references


class DocumentationOutline(TypedDict, total=False):
    """Generated documentation outline based on template"""
    template_id: str
    template_structure: Dict[str, Any]
    section_mappings: List[SectionMapping]
    unmapped_sections: List[str]
    outline_metadata: Dict[str, Any]


class SectionDraft(TypedDict, total=False):
    """Drafted content for a documentation section"""
    section_id: str
    template_section_id: Optional[str]
    heading: str
    content: str
    metadata: Dict[str, Any]
    order: int
    status: Literal["drafted", "reviewed", "final"]


class AgentConfig(TypedDict, total=False):
    """Configuration for the model documentation agent"""
    codebase: Dict[str, Any]  # file_extensions, exclude_patterns
    llm: Dict[str, Any]  # model, temperature
    template: Dict[str, Any]  # template_id, default_path
    output: Dict[str, Any]  # base_dir, create_timestamped_dir


class AgentState(TypedDict, total=False):
    """Complete state for the model documentation workflow"""
    codebase_id: str
    codebase_path: str
    config: AgentConfig
    
    # Phase 1: Codebase Discovery
    file_list: List[FileMetadata]
    codebase_metadata: Optional[CodebaseMetadata]
    file_hierarchy: Optional[Dict[str, Any]]
    code_stats: Optional[Dict[str, Any]]
    
    # Phase 2: Summarization
    file_summaries: Dict[str, str]  # file_path -> summary
    hierarchical_summary: Optional[HierarchicalSummary]
    
    # Phase 3: Outline Generation
    documentation_outline: Optional[DocumentationOutline]
    template: Optional[Dict[str, Any]]
    
    # Phase 4: Content Generation
    section_drafts: List[SectionDraft]
    
    # Phase 5: Assembly
    final_documentation: Optional[str]
    final_documentation_path: Optional[str]
    output_directory: Optional[str]
    
    # Workflow tracking
    last_node: Optional[str]
    logs: List[str]
    phase_stats: Dict[str, Any]
    phase_reports: Dict[str, Any]


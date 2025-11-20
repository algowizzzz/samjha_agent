from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

PhaseStatus = Literal["pending", "running", "success", "failed"]


class PageEntry(TypedDict, total=False):
    page_num: int
    text: str


class HeadingEntry(TypedDict, total=False):
    level: str  # "H1"..."H5"
    title: str
    page: Optional[int]
    numbering: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    line_number: Optional[int]


class TocEntry(TypedDict, total=False):
    title: str
    level: int
    page_number: Optional[int]
    notes: Optional[str]
    confidence: Optional[str]


class ImageMetadata(TypedDict, total=False):
    image_id: str
    path: str
    page_number: Optional[int]
    caption: Optional[str]


class DocMeta(TypedDict, total=False):
    doc_title: str
    doc_source: Literal["upload", "sharepoint", "s3", "unknown"]
    source_path: str
    file_type: Optional[str]
    page_count: int
    word_count: Optional[int]
    md_file_id: Optional[str]
    md_path: Optional[str]
    version: int


class StructureData(TypedDict, total=False):
    raw_text: str
    pages: List[PageEntry]
    headings: List[HeadingEntry]
    toc_detected: bool
    toc_entries: List[TocEntry]
    images: List[ImageMetadata]


class TemplateMeta(TypedDict, total=False):
    template_id: str
    template_label: Optional[str]
    template_text: Optional[str]
    template_categories: List[str]
    max_section_words: int


class DocSummaryReport(TypedDict, total=False):
    summary: str
    document_type: Optional[str]
    purpose: Optional[str]
    audience: Optional[str]
    themes: List[str]
    confidence: Literal["high", "medium", "low"]


class TocReviewReport(TypedDict, total=False):
    toc_present: bool
    toc_label: Optional[str]
    structure_score: Literal["excellent", "good", "fair", "poor"]
    entries: List[TocEntry]
    observations: List[str]
    gaps: List[str]


class TemplateCategoryAssessment(TypedDict, total=False):
    name: str
    coverage: Literal["complete", "partial", "missing"]
    effort: Literal["none", "low", "medium", "high"]
    gaps: List[str]
    actions: List[str]


class TemplateFitnessSummary(TypedDict, total=False):
    template_id: str
    template_label: Optional[str]
    overall_alignment: Literal["excellent", "good", "fair", "poor", "unknown"]
    categories: List[TemplateCategoryAssessment]
    narrative: str


class SectionStrategyReport(TypedDict, total=False):
    verdict: Literal["ready", "needs_improvement"]
    rationale: str
    recommended_section_level: Literal["h1", "h2", "h3", "h4", "h5"]
    fallback_levels: List[str]
    estimated_sections: Optional[int]
    next_steps: List[str]


class SectionChunk(TypedDict, total=False):
    section_title: str
    method: Literal["headings", "toc", "merged"]
    page_range: List[int]
    char_range: Optional[List[int]]
    boundary_check: Literal["perfect", "ok", "incomplete", "unknown"]
    issues: List[str]
    text: str


class SuggestedChange(TypedDict, total=False):
    id: str
    index: int
    section_title: str
    page_hint: Optional[int]
    location_instruction: Optional[str]
    original_text: str
    suggested_text: str
    severity: Literal["low", "medium", "high"]
    type: Literal[
        "grammar",
        "clarity",
        "structural",
        "missing_content",
        "terminology",
        "tone",
        "compliance_precision",
    ]
    reason: str
    status: Literal["pending", "applied", "ignored"]


class SectionReview(TypedDict, total=False):
    section_title: str
    fit: Literal["none", "partial", "good"]
    severity: Literal["low", "medium", "high"]
    issues: List[SuggestedChange]
    improvement_guidance: List[str]


class Phase2SummaryReport(TypedDict, total=False):
    overall_posture: Literal["ready", "needs_work", "needs_overhaul"]
    section_heatmap: Dict[str, Literal["low", "medium", "high"]]
    systemic_gaps: List[str]
    narrative: str
    total_issues: int
    high_severity_count: int


class ChangeSelectionPlan(TypedDict, total=False):
    apply_mode: Literal["all", "by_ids", "by_severity", "by_section"]
    change_ids_to_apply: List[str]
    rationale: str


class Phase1Data(TypedDict, total=False):
    toc_review: Optional[TocReviewReport]


class Phase2Data(TypedDict, total=False):
    check_results: Dict[str, Any]  # Results from 4 holistic checks
    all_issues: List[Dict[str, Any]]  # Consolidated issues from all checks
    synthesis_report: Optional[Dict[str, Any]]  # Synthesis summary


class VfsArtifact(TypedDict, total=False):
    path: str
    label: str
    last_updated: Optional[str]


class AgentState(TypedDict, total=False):
    run_id: str
    doc_id: str
    control: Optional[str]
    last_node: Optional[str]
    errors: List[str]
    phase1_status: PhaseStatus
    phase2_status: PhaseStatus
    locked_by: Optional[str]
    lock_timestamp: Optional[str]
    doc_meta: DocMeta
    structure: StructureData
    template_meta: TemplateMeta
    phase1: Phase1Data
    phase2: Phase2Data
    file_metadata: Optional[Dict[str, Any]]
    vfs_artifacts: List[VfsArtifact]
    logs: List[str]
    agent_transcript: List[Dict[str, Any]]


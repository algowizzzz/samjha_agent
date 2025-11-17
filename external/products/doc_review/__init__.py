"""
Document Review Product

Provides document review and analysis capabilities with risk assessment.
"""

from external.products.doc_review.agent import DocReviewAgent
from external.products.doc_review.store import DocReviewStore
from external.products.doc_review.vfs import DocReviewVFSAdapter

__all__ = [
    "DocReviewAgent",
    "DocReviewStore",
    "DocReviewVFSAdapter",
]


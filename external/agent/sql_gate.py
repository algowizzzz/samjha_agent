"""
SQL generation gate - ensures QuerySpec is complete before SQL generation.
"""

import re
from typing import Optional


def get_domain_bool(domain_md: str, key: str) -> Optional[bool]:
    """
    Extract boolean value from domain.md markdown.
    
    Matches patterns like: "- listing_allows_empty_metrics: true"
    
    Args:
        domain_md: Domain markdown content
        key: Key to search for
        
    Returns:
        Boolean value or None if not found
    """
    pattern = rf"(?im)^\s*-\s*{re.escape(key)}\s*:\s*(true|false)\s*$"
    match = re.search(pattern, domain_md)
    if not match:
        return None
    return match.group(1).lower() == "true"


def metrics_empty_allowed(query_spec: dict, domain_md: str) -> bool:
    """
    Check if empty metrics are allowed for this query.
    
    Rules:
    - Domain must have listing_allows_empty_metrics: true
    - query_spec.metrics must be empty list
    - aggregation_plan must contain "listing"
    
    Args:
        query_spec: Query spec dictionary
        domain_md: Domain markdown content
        
    Returns:
        True if empty metrics are allowed
    """
    allow = get_domain_bool(domain_md, "listing_allows_empty_metrics")
    if allow is not True:
        return False
    
    metrics = query_spec.get("metrics", [])
    aggregation_plan = query_spec.get("aggregation_plan", "").lower()
    
    return (
        isinstance(metrics, list)
        and len(metrics) == 0
        and "listing" in aggregation_plan
    )


def spec_ready_for_sql(query_spec: dict, query_spec_status: dict, domain_md: str) -> bool:
    """
    Check if QuerySpec is ready for SQL generation.
    
    Required fields must be verified/defaulted (not missing/conflict):
    - business_question
    - start_table_grain (start_table + grain)
    - time (unless query_spec.time.rule == "no_time")
    - metrics (unless listing_allows_empty_metrics == true)
    
    Args:
        query_spec: Query spec dictionary
        query_spec_status: Query spec status dictionary
        domain_md: Domain markdown content
        
    Returns:
        True if spec is ready for SQL generation
    """
    # Check business_question
    business_question_status = query_spec_status.get("business_question", {})
    if business_question_status.get("status") in ["missing", "conflict"]:
        return False
    if business_question_status.get("blocks_execution", False):
        return False
    
    # Check start_table_grain
    start_table_grain_status = query_spec_status.get("start_table_grain", {})
    if start_table_grain_status.get("status") in ["missing", "conflict"]:
        return False
    if start_table_grain_status.get("blocks_execution", False):
        return False
    
    # Check time (unless no_time rule)
    time_rule = query_spec.get("time", {}).get("rule", "")
    if time_rule != "no_time":
        time_status = query_spec_status.get("time", {})
        if time_status.get("status") in ["missing", "conflict"]:
            return False
        if time_status.get("blocks_execution", False):
            return False
    
    # Check metrics (unless listing allows empty)
    metrics_status = query_spec_status.get("metrics", {})
    if metrics_status.get("status") in ["missing", "conflict"]:
        # Check if empty metrics are allowed
        if not metrics_empty_allowed(query_spec, domain_md):
            return False
    if metrics_status.get("blocks_execution", False):
        # Even if empty allowed, if it blocks execution, not ready
        if not metrics_empty_allowed(query_spec, domain_md):
            return False
    
    return True


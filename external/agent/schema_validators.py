"""
Schema validators for Parquet Agent contracts.
Validates DeciderOutput, ExecutorReport, QuerySpec, and InvestigationPlan against JSON schemas.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

try:
    import jsonschema
    from jsonschema import validate, ValidationError, RefResolver
except ImportError:
    raise ImportError("jsonschema library required. Install with: pip install jsonschema")


# Schema directory
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def _load_schema(schema_name: str) -> dict:
    """Load a JSON schema file."""
    schema_path = SCHEMAS_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r') as f:
        return json.load(f)


def _get_resolver() -> RefResolver:
    """Create a resolver for $ref resolution."""
    store = {}
    for schema_file in SCHEMAS_DIR.glob("*.json"):
        with open(schema_file, 'r') as f:
            schema = json.load(f)
            schema_id = schema.get("$id", schema_file.stem)
            store[schema_id] = schema
    
    return RefResolver.from_schema({"$id": "root"}, store=store)


def validate_decider_output(output: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate DeciderOutput against decider_output.schema.json.
    
    Args:
        output: Decider output dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema = _load_schema("decider_output.schema.json")
        resolver = _get_resolver()
        validate(instance=output, schema=schema, resolver=resolver)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_executor_report(report: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate ExecutorReport against executor_report.schema.json.
    
    Args:
        report: Executor report dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema = _load_schema("executor_report.schema.json")
        resolver = _get_resolver()
        validate(instance=report, schema=schema, resolver=resolver)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_query_spec(spec: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate QuerySpec against query_spec.schema.json.
    
    Args:
        spec: Query spec dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema = _load_schema("query_spec.schema.json")
        resolver = _get_resolver()
        validate(instance=spec, schema=schema, resolver=resolver)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_query_spec_status(status: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate QuerySpecStatus against query_spec_status.schema.json.
    
    Args:
        status: Query spec status dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema = _load_schema("query_spec_status.schema.json")
        resolver = _get_resolver()
        validate(instance=status, schema=schema, resolver=resolver)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_investigation_plan(plan: list) -> Tuple[bool, Optional[str]]:
    """
    Validate InvestigationPlan (list of steps) against investigation_plan_step.schema.json.
    
    Args:
        plan: List of investigation plan steps
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(plan, list):
        return False, "Investigation plan must be a list"
    
    # Keep validator aligned with schema (decider_output.schema.json allows up to 6).
    if len(plan) > 6:
        return False, f"Investigation plan has {len(plan)} steps, maximum is 6"
    
    try:
        schema = _load_schema("investigation_plan_step.schema.json")
        resolver = _get_resolver()
        
        for i, step in enumerate(plan):
            validate(instance=step, schema=schema, resolver=resolver)
            
            # Validate step number matches index
            if step.get("step") != i + 1:
                return False, f"Step {i+1} has incorrect step number: {step.get('step')}"
        
        return True, None
    except ValidationError as e:
        return False, f"Validation error at step {i+1}: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_policy_limits(limits: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate PolicyLimits against policy_limits.schema.json.
    
    Args:
        limits: Policy limits dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema = _load_schema("policy_limits.schema.json")
        resolver = _get_resolver()
        validate(instance=limits, schema=schema, resolver=resolver)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


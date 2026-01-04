"""
Chain validation utilities (extracted from legacy service).
"""
from typing import Dict, List, Tuple, Optional


def validate_chain_structure(steps: List[Dict]) -> Tuple[bool, Optional[str]]:
    """
    Validate chain structure:
    - At least 1 step required
    - Each step must have required_inputs (array) and prompt (non-empty)
    - Inputs must be valid: R0 always valid, R1 only if step 1 exists, etc.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not steps or len(steps) == 0:
        return False, "At least one step is required"
    
    # Validate each step
    for i, step in enumerate(steps):
        step_index = i + 1  # Step numbers are 1-based
        required_inputs = step.get("required_inputs", [])
        prompt = step.get("prompt", "")
        
        # Check required_inputs is an array
        if not isinstance(required_inputs, list):
            return False, f"Step {step_index}: required_inputs must be an array"
        
        # Check at least one input
        if len(required_inputs) == 0:
            return False, f"Step {step_index}: At least one required_input is required"
        
        # Check prompt is non-empty
        if not prompt or not prompt.strip():
            return False, f"Step {step_index}: Prompt is required and cannot be empty"
        
        # Validate inputs: R0 always valid, R1 only if step 1 exists, etc.
        available_inputs = ["R0"]  # R0 always available
        for j in range(1, step_index):  # R1..R(N-1) available
            available_inputs.append(f"R{j}")
        
        invalid_inputs = [inp for inp in required_inputs if inp not in available_inputs]
        if invalid_inputs:
            return False, f"Step {step_index}: Invalid inputs {invalid_inputs}. Available: {available_inputs}"
    
    return True, None


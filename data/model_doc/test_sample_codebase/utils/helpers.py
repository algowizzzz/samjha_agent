"""
Helper utility functions.
"""


def format_number(num: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{num:.{decimals}f}"


def validate_input(value: str, min_val: float = None, max_val: float = None) -> float:
    """Validate and convert input string to float.
    
    Args:
        value: Input string
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Converted float value
        
    Raises:
        ValueError: If value is invalid or out of range
    """
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            raise ValueError(f"Value {num} is below minimum {min_val}")
        if max_val is not None and num > max_val:
            raise ValueError(f"Value {num} is above maximum {max_val}")
        return num
    except ValueError as e:
        if "could not convert" in str(e).lower():
            raise ValueError(f"Invalid number: {value}")
        raise

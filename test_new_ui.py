#!/usr/bin/env python3
"""Quick test script to verify the new UI template loads correctly"""

import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def test_template():
    """Test if the template can be loaded and rendered"""
    try:
        template_dir = Path("web/templates")
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Test loading the template
        print("1. Testing template load...")
        template = env.get_template("doc_review_cockpit_new.html")
        print("   ✓ Template loaded successfully")
        
        # Test rendering with minimal context
        print("2. Testing template rendering...")
        rendered = template.render(
            url_for=lambda x, **kwargs: f"/{x}",
        )
        print(f"   ✓ Template rendered successfully ({len(rendered)} characters)")
        
        # Check for common issues
        print("3. Checking for common issues...")
        issues = []
        
        if "doc_review_cockpit.css" not in rendered:
            issues.append("CSS file reference not found")
        
        if "<script>" not in rendered or "</script>" not in rendered:
            issues.append("JavaScript block missing")
        
        if "doc-review-container" not in rendered:
            issues.append("Main container class not found")
        
        if issues:
            print(f"   ⚠ Found {len(issues)} potential issues:")
            for issue in issues:
                print(f"     - {issue}")
        else:
            print("   ✓ No obvious issues found")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_template()
    sys.exit(0 if success else 1)


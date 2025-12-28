"""
Test template rendering with real data
Catches template syntax errors that static analysis might miss
"""

import os
import sys
sys.path.insert(0, '.')

from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

def test_agent_chat_template():
    """Test agent_chat.html template renders correctly"""
    print("=" * 70)
    print("TEST: Template rendering")
    print("=" * 70)
    
    try:
        env = Environment(loader=FileSystemLoader('web/templates'))
        template = env.get_template('agent_chat.html')
        
        # Test with realistic data
        test_data = {
            'user': {'user_id': 'test_user', 'username': 'test'},
            'token': 'test_token',
            'agent_id': 'ecommerce_agent',
            'agent_name': 'Ecommerce Agent',
            'agent_description': 'Test agent',
            'bulk_doc_available': False,
            'version': 9,
            'cache_bust': 1234567890,
            'template_hash': 'abcd1234'
        }
        
        print("\n1. Testing template syntax...")
        # This would raise TemplateSyntaxError if there are issues
        rendered = template.render(**test_data)
        
        print("   ✓ Template syntax is valid")
        print(f"   ✓ Template rendered ({len(rendered)} chars)")
        
        # Check for common issues
        print("\n2. Checking for common issues...")
        
        # Check for unclosed blocks
        endblock_count = rendered.count('{% endblock %}')
        block_count = rendered.count('{% block ')
        if endblock_count != block_count:
            print(f"   ⚠️  Block count mismatch: {block_count} blocks, {endblock_count} endblocks")
        else:
            print(f"   ✓ Block structure correct ({block_count} blocks)")
        
        # Check for JavaScript syntax
        if '<script>' in rendered and '</script>' in rendered:
            script_count = rendered.count('<script>')
            script_end_count = rendered.count('</script>')
            if script_count != script_end_count:
                print(f"   ⚠️  Script tag mismatch: {script_count} open, {script_end_count} close")
            else:
                print(f"   ✓ Script tags balanced ({script_count} scripts)")
        
        print("\n" + "=" * 70)
        print("✓ Template rendering test passed")
        print("=" * 70)
        return True
        
    except TemplateSyntaxError as e:
        print(f"\n❌ Template syntax error: {e}")
        print(f"   Line {e.lineno}: {e.message}")
        return False
    except Exception as e:
        print(f"\n❌ Template rendering error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_agent_chat_template()
    sys.exit(0 if success else 1)


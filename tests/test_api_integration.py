"""
Quick integration test script - can be run directly
Tests the actual API endpoints with real database
"""

import os
import sys
import json
sys.path.insert(0, '.')

os.environ['DATABASE_URL'] = 'postgresql+psycopg2://saadahmed@localhost:5432/samjha_agent'

def test_conversation_id_handling():
    """Test conversation_id handling with session IDs"""
    from core.db.session import get_db_session
    from external.agent.persistence import get_or_create_conversation
    
    print("=" * 70)
    print("TEST: conversation_id handling (frontend session IDs)")
    print("=" * 70)
    
    test_cases = [
        ("sess-4ynvmphf", "Frontend session ID"),
        (None, "No conversation_id"),
        ("00000000-0000-0000-0000-000000000000", "Invalid UUID"),
    ]
    
    for conversation_id, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Input: {conversation_id}")
        
        with get_db_session() as db:
            result_id = get_or_create_conversation(db, conversation_id, "test_user", "ecommerce_agent")
            db.commit()
        
        print(f"  Output: {result_id}")
        
        # Validate
        assert len(result_id) == 36, "Should be UUID (36 chars)"
        assert result_id != conversation_id or conversation_id is None, "Should not be same as input session ID"
        
        # Verify in DB
        with get_db_session() as db:
            from core.db.models import Conversation
            conv = db.get(Conversation, result_id)
            assert conv is not None, "Conversation should exist in DB"
            print(f"  ✓ Conversation exists in DB")
    
    print("\n" + "=" * 70)
    print("✓ All conversation_id tests passed")
    print("=" * 70)


def test_message_foreign_key():
    """Test message creation doesn't violate foreign key constraint"""
    from core.db.session import get_db_session
    from external.agent.persistence import get_or_create_conversation, append_message
    from core.db.models import Message
    
    print("\n" + "=" * 70)
    print("TEST: Message foreign key constraint")
    print("=" * 70)
    
    # Use session ID (what frontend sends)
    session_id = "sess-test-fk-123"
    
    print(f"\n1. Create/get conversation with session ID: {session_id}")
    with get_db_session() as db:
        conv_id = get_or_create_conversation(db, session_id, "test_user", "ecommerce_agent")
        db.commit()
    print(f"   Conversation ID: {conv_id}")
    
    print(f"\n2. Append message (should not violate FK constraint)")
    with get_db_session() as db:
        msg_id = append_message(db, conv_id, "user", "Test query")
        db.commit()
    print(f"   Message ID: {msg_id}")
    
    print(f"\n3. Verify message exists")
    with get_db_session() as db:
        msg = db.get(Message, msg_id)
        assert msg is not None, "Message should exist"
        assert msg.conversation_id == conv_id, "Should be linked to conversation"
        print(f"   ✓ Message exists and is properly linked")
    
    print("\n" + "=" * 70)
    print("✓ Foreign key constraint test passed")
    print("=" * 70)


if __name__ == '__main__':
    try:
        test_conversation_id_handling()
        test_message_foreign_key()
        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


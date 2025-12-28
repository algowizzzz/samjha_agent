"""
Integration tests for Agent Run Routes - End-to-end API testing
Tests actual HTTP endpoints with real database interactions
"""

import unittest
import json
from datetime import datetime

from web.app import create_app
from core.db.session import get_db_session, get_engine
from core.db.models import Base, Conversation, Message, Run
from external.agent.persistence import ensure_schema, create_agent_db


class TestAgentRunRoutesIntegration(unittest.TestCase):
    """End-to-end integration tests for agent run routes"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and Flask app"""
        # Use test database
        import os
        os.environ['DATABASE_URL'] = 'postgresql+psycopg2://saadahmed@localhost:5432/samjha_agent_test'
        
        # Create test database schema
        ensure_schema()
        
        # Create Flask app
        cls.app, cls.socketio = create_app()
        cls.client = cls.app.test_client()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        
        # Create test agent
        with get_db_session() as db:
            agent = create_agent_db(
                db, 
                agent_id="test_agent",
                name="Test Agent",
                agent_type="structured",
                description="Test agent",
                domain_file="ecommerce_agent_domain.md",
                data_folder="ECommerce"
            )
            db.commit()
    
    def setUp(self):
        """Set up for each test"""
        # Create test session (simulate login)
        with self.client.session_transaction() as sess:
            sess['token'] = 'test_token_123'
            sess['user_id'] = 'test_user'
    
    def test_start_run_with_session_id_conversation_id(self):
        """Test: Frontend sends session ID (sess-xxx) as conversation_id"""
        # This simulates what the frontend actually sends
        response = self.client.post(
            '/api/agents/test_agent/runs',
            json={
                'query': 'What are the top 5 products?',
                'conversation_id': 'sess-4ynvmphf',  # Session ID from frontend
                'show_thinking': False
            },
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Should return valid run_id and conversation_id (UUID, not session ID)
        self.assertIn('run_id', data)
        self.assertIn('conversation_id', data)
        
        # conversation_id should be a UUID (36 chars with dashes)
        conv_id = data['conversation_id']
        self.assertEqual(len(conv_id), 36)
        self.assertNotEqual(conv_id, 'sess-4ynvmphf')  # Should NOT be the session ID
        
        # Verify conversation was created in DB
        with get_db_session() as db:
            conv = db.get(Conversation, conv_id)
            self.assertIsNotNone(conv)
            self.assertEqual(conv.user_id, 'test_user')
            self.assertEqual(conv.agent_id, 'test_agent')
    
    def test_start_run_with_none_conversation_id(self):
        """Test: No conversation_id provided"""
        response = self.client.post(
            '/api/agents/test_agent/runs',
            json={
                'query': 'What are the top 5 products?',
                'show_thinking': False
            },
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('conversation_id', data)
        conv_id = data['conversation_id']
        self.assertEqual(len(conv_id), 36)  # Should be UUID
    
    def test_start_run_with_valid_uuid_conversation_id(self):
        """Test: Valid UUID conversation_id provided (existing conversation)"""
        # First create a conversation
        with get_db_session() as db:
            from external.agent.persistence import create_conversation
            conv_id = create_conversation(db, 'test_user', 'test_agent')
            db.commit()
        
        # Now use it
        response = self.client.post(
            '/api/agents/test_agent/runs',
            json={
                'query': 'What are the top 5 products?',
                'conversation_id': conv_id,  # Valid UUID
                'show_thinking': False
            },
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Should use the same conversation_id
        self.assertEqual(data['conversation_id'], conv_id)
    
    def test_start_run_with_invalid_uuid_conversation_id(self):
        """Test: Invalid UUID provided (doesn't exist in DB)"""
        fake_uuid = '00000000-0000-0000-0000-000000000000'
        
        response = self.client.post(
            '/api/agents/test_agent/runs',
            json={
                'query': 'What are the top 5 products?',
                'conversation_id': fake_uuid,
                'show_thinking': False
            },
            content_type='application/json'
        )
        
        # Should succeed and create new conversation
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Should return a different conversation_id (new one created)
        new_conv_id = data['conversation_id']
        self.assertNotEqual(new_conv_id, fake_uuid)
        self.assertEqual(len(new_conv_id), 36)
    
    def test_message_foreign_key_constraint(self):
        """Test: Messages are properly linked to conversations (no FK violation)"""
        from external.agent.persistence import get_or_create_conversation, append_message
        
        with get_db_session() as db:
            # Simulate frontend session ID
            session_id = "sess-test123"
            conv_id = get_or_create_conversation(db, session_id, 'test_user', 'test_agent')
            db.commit()
        
        # Should be able to append message without FK violation
        with get_db_session() as db:
            msg_id = append_message(db, conv_id, 'user', 'Test message')
            db.commit()
            self.assertIsNotNone(msg_id)
        
        # Verify message exists
        with get_db_session() as db:
            msg = db.get(Message, msg_id)
            self.assertIsNotNone(msg)
            self.assertEqual(msg.conversation_id, conv_id)
            self.assertEqual(msg.content, 'Test message')
    
    def test_missing_query(self):
        """Test: Missing query parameter"""
        response = self.client.post(
            '/api/agents/test_agent/runs',
            json={
                'conversation_id': 'sess-xxx',
                'show_thinking': False
            },
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('query', data['error'].lower())
    
    def test_nonexistent_agent(self):
        """Test: Agent doesn't exist"""
        response = self.client.post(
            '/api/agents/nonexistent_agent/runs',
            json={
                'query': 'Test query',
                'show_thinking': False
            },
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        # Optionally clean up test data
        pass


if __name__ == '__main__':
    unittest.main()


"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Agent Routes for SAJHA MCP Server - Moved to external/
"""

import logging
from flask import render_template, make_response
from pathlib import Path
import os
import time
import hashlib

from external.routes.base_routes import BaseRoutes


class AgentRoutes(BaseRoutes):
    """Agent-related routes - moved to external/"""

    def __init__(self, auth_manager, tools_registry):
        """Initialize agent routes"""
        super().__init__(auth_manager, tools_registry)

    def register_routes(self, app):
        """Register agent routes"""

        @app.route('/agents')
        @self.login_required
        def agents_selection():
            """Agent selection page showing all available agents grouped by type."""
            user_session = self.get_user_session()
            cache_bust_value = int(time.time())
            response = make_response(render_template('agents.html', user=user_session, cache_bust=cache_bust_value))
            # Add aggressive cache-busting headers to prevent browser caching
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        @app.route('/agent/chat')
        @app.route('/agent/chat/<agent_id>')
        @self.login_required
        def agent_chat(agent_id=None):
            """Chat UI for a specific agent instance."""
            user_session = self.get_user_session()
            
            # If no agent_id provided, redirect to agents selection
            if not agent_id:
                from flask import redirect, url_for
                return redirect(url_for('agents_selection'))
            
            # Validate agent exists
            try:
                from external.core.db.session import get_db_session
                from external.agent.persistence import get_agent_db
                
                with get_db_session() as db:
                    agent = get_agent_db(db, agent_id)
                if not agent:
                    return render_template('error.html', 
                                         user=user_session,
                                         error="Agent Not Found",
                                         message=f"Agent '{agent_id}' does not exist"), 404
            except Exception as e:
                logging.error(f"Error loading agent {agent_id}: {e}")
                return render_template('error.html',
                                     user=user_session,
                                     error="Error",
                                     message=f"Failed to load agent: {str(e)}"), 500
            
            cache_bust_value = int(time.time())
            
            # Read the actual template file to get its hash for cache busting
            try:
                template_path = os.path.join('web', 'templates', 'agent_chat.html')
                if os.path.exists(template_path):
                    with open(template_path, 'rb') as f:
                        template_hash = hashlib.md5(f.read()).hexdigest()[:8]
                else:
                    template_hash = str(cache_bust_value)
            except:
                template_hash = str(cache_bust_value)
            
            # Add version 8 to template, pass agent info
            response = make_response(render_template('agent_chat.html', 
                                                   user=user_session, 
                                                   token=user_session.get('token') if user_session else None,
                                                   agent_id=agent_id,
                                                   agent_name=agent.get('name', ''),
                                                   agent_description=agent.get('description', ''),
                                                   cache_bust=cache_bust_value, 
                                                   version=8, 
                                                   template_hash=template_hash))
            # Add aggressive cache-busting headers to prevent browser caching
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['Last-Modified'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
            response.headers['ETag'] = f'"v8-{template_hash}-{cache_bust_value}"'
            response.headers['Vary'] = 'Cache-Control'
            return response


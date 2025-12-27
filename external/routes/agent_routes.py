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

from routes.base_routes import BaseRoutes


class AgentRoutes(BaseRoutes):
    """Agent-related routes - moved to external/"""

    def __init__(self, auth_manager, tools_registry):
        """Initialize agent routes"""
        super().__init__(auth_manager, tools_registry)

    def register_routes(self, app):
        """Register agent routes"""

        @app.route('/agent/chat')
        @self.login_required
        def agent_chat():
            """Simple chat UI for parquet_agent showing response and prompt monitor."""
            user_session = self.get_user_session()
            # Ensure the agent tool exists; if not, page still renders but calls will fail gracefully
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
            
            # Add version 8 to template
            response = make_response(render_template('agent_chat.html', user=user_session, token=user_session.get('token') if user_session else None, cache_bust=cache_bust_value, version=8, template_hash=template_hash))
            # Add aggressive cache-busting headers to prevent browser caching
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['Last-Modified'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
            response.headers['ETag'] = f'"v8-{template_hash}-{cache_bust_value}"'  # Add version and hash to ETag for cache busting
            response.headers['Vary'] = 'Cache-Control'
            return response


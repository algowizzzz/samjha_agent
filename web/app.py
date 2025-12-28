"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Flask application for SAJHA MCP Server - Refactored with modular routes
"""

import logging
import os
from flask import Flask, render_template, jsonify, redirect, url_for, session
from flask_socketio import SocketIO
from flask_cors import CORS
from datetime import timedelta, datetime

# Import core modules
from core.auth_manager import AuthManager
from core.mcp_handler import MCPHandler
from tools.tools_registry import ToolsRegistry

# Import route modules
from routes import (
    AuthRoutes,
    DashboardRoutes,
    ToolsRoutes,
    AdminRoutes,
    MonitoringRoutes,
    ApiRoutes,
    SocketIOHandlers
)

# AI Bulk Doc Analysis (isolated under external/)
try:
    from external.ai_bulk_doc_analysis.blueprint import create_bulk_doc_blueprint
    BULK_DOC_AVAILABLE = True
except Exception as e:
    logging.warning(f"Bulk doc analysis feature not available: {e}")
    create_bulk_doc_blueprint = None
    BULK_DOC_AVAILABLE = False

# Import external agent routes (optional)
try:
    from external.routes.agent_routes import AgentRoutes
    AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Agent routes not available: {e}. Server will run without agent features.")
    AgentRoutes = None
    AGENT_AVAILABLE = False

# Global instances
app = None
socketio = None
auth_manager = None
mcp_handler = None
tools_registry = None


def create_app():
    """Create and configure Flask application"""
    global app, socketio, auth_manager, mcp_handler, tools_registry
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'sajha-mcp-server-secret-key-2025'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
    
    # Enable CORS with credentials support
    CORS(app, 
         supports_credentials=True,
         origins=["http://localhost:3003", "http://localhost:3002", "http://localhost:3001", "http://localhost:3000", "http://127.0.0.1:3003", "http://127.0.0.1:3002", "http://127.0.0.1:3001", "http://127.0.0.1:3000"],
         allow_headers=["Content-Type", "Authorization", "X-API-Key"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize managers
    auth_manager = AuthManager()
    tools_registry = ToolsRegistry()
    mcp_handler = MCPHandler(tools_registry=tools_registry, auth_manager=auth_manager)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize DB schema and import existing data
    try:
        from external.agent.persistence import ensure_schema, import_prompts_from_files, import_agents_from_files
        from core.db.session import get_db_session
        ensure_schema()
        with get_db_session() as db:
            prompts_imported = import_prompts_from_files(db)
            agents_imported = import_agents_from_files(db)
            db.commit()
            if prompts_imported > 0 or agents_imported > 0:
                logging.info(f"Imported {prompts_imported} prompts and {agents_imported} agents from files to DB")
    except Exception as e:
        logging.warning(f"DB initialization failed (continuing anyway): {e}")
    
    # Register routes
    register_all_routes(app, socketio)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register health check
    register_health_check(app)
    
    return app, socketio


def register_all_routes(app, socketio):
    """Register all route modules"""

    # Always register auth + api (needed for /api/tools/execute and session login)
    auth_routes = AuthRoutes(auth_manager)
    api_routes = ApiRoutes(auth_manager, tools_registry, mcp_handler)
    
    # Home page with 3 boxes
    @app.route("/")
    def index():
        # Check if user is logged in
        if 'token' not in session:
            return redirect(url_for('login'))
        session_data = auth_manager.validate_session(session.get('token'))
        if not session_data:
            session.pop('token', None)
            return redirect(url_for('login'))
        return render_template('home.html', user=session_data, bulk_doc_available=BULK_DOC_AVAILABLE)

    # Register AI Bulk Doc Analysis UI (isolated feature)
    if BULK_DOC_AVAILABLE and create_bulk_doc_blueprint is not None:
        try:
            app.register_blueprint(create_bulk_doc_blueprint(auth_manager))
            logging.info("Bulk doc analysis blueprint registered successfully")
        except Exception as e:
            logging.error(f"Failed to register bulk doc analysis blueprint: {e}")

    # Register routes
    auth_routes.register_routes(app)
    api_routes.register_routes(app)

    # Register MCP dashboard and management routes
    dashboard_routes = DashboardRoutes(auth_manager, tools_registry)
    tools_routes = ToolsRoutes(auth_manager, tools_registry)
    admin_routes = AdminRoutes(auth_manager, tools_registry)
    monitoring_routes = MonitoringRoutes(auth_manager, tools_registry)
    socketio_handlers = SocketIOHandlers(socketio, auth_manager, tools_registry, mcp_handler)

    dashboard_routes.register_routes(app)
    tools_routes.register_routes(app)
    admin_routes.register_routes(app)
    monitoring_routes.register_routes(app)
    socketio_handlers.register_handlers()
    
    # Register external agent routes
    if AGENT_AVAILABLE and AgentRoutes is not None:
        try:
            agent_routes = AgentRoutes(auth_manager, tools_registry)
            agent_routes.register_routes(app)
            logging.info("Agent routes registered successfully")
            
            # Register agent run routes (SSE)
            from external.routes.agent_run_routes import AgentRunRoutes
            agent_run_routes = AgentRunRoutes(auth_manager, tools_registry)
            agent_run_routes.register_routes(app)
            logging.info("Agent run routes (SSE) registered successfully")
        except Exception as e:
            logging.error(f"Failed to register agent routes: {e}")
    else:
        logging.info("Agent features not available - server running in base mode")

    logging.info("All routes registered successfully")


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found(error):
        """404 error handler"""
        return render_template('error.html',
                             error="Page Not Found",
                             message="The requested page does not exist"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500 error handler"""
        return render_template('error.html',
                             error="Internal Server Error",
                             message="An unexpected error occurred"), 500


def register_health_check(app):
    """Register health check endpoint"""
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })


if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, host='0.0.0.0', port=5555, debug=True, allow_unsafe_werkzeug=True)

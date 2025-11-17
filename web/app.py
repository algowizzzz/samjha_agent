"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Flask application for SAJHA MCP Server - Refactored with modular routes
"""

import logging
from flask import Flask, render_template, jsonify
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

# Import external agent routes (optional)
try:
    from external.routes.agent_routes import AgentRoutes
    from external.routes.agent_socketio_handlers import AgentSocketIOHandlers
    AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Agent routes not available: {e}. Server will run without agent features.")
    AgentRoutes = None
    AgentSocketIOHandlers = None
    AGENT_AVAILABLE = False

# Import document review routes (optional)
try:
    from external.routes.doc_review_routes import DocReviewRoutes
    DOC_REVIEW_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Doc review routes not available: {e}.")
    DocReviewRoutes = None
    DOC_REVIEW_AVAILABLE = False

# Import model documentation routes (optional)
try:
    from external.routes.model_doc_routes import ModelDocRoutes
    MODEL_DOC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model doc routes not available: {e}.")
    ModelDocRoutes = None
    MODEL_DOC_AVAILABLE = False

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
    
    # Enable CORS
    CORS(app)
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize managers
    auth_manager = AuthManager()
    tools_registry = ToolsRegistry()
    mcp_handler = MCPHandler(tools_registry=tools_registry, auth_manager=auth_manager)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Register routes
    register_all_routes(app, socketio)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register health check
    register_health_check(app)
    
    return app, socketio


def register_all_routes(app, socketio):
    """Register all route modules"""
    
    # Initialize route classes
    auth_routes = AuthRoutes(auth_manager)
    dashboard_routes = DashboardRoutes(auth_manager, tools_registry)
    tools_routes = ToolsRoutes(auth_manager, tools_registry)
    admin_routes = AdminRoutes(auth_manager, tools_registry)
    monitoring_routes = MonitoringRoutes(auth_manager, tools_registry)
    api_routes = ApiRoutes(auth_manager, tools_registry, mcp_handler)
    socketio_handlers = SocketIOHandlers(socketio, auth_manager, tools_registry, mcp_handler)
    
    # Register blueprints
    auth_routes.register_routes(app)
    dashboard_routes.register_routes(app)
    tools_routes.register_routes(app)
    admin_routes.register_routes(app)
    monitoring_routes.register_routes(app)
    api_routes.register_routes(app)
    
    # Register SocketIO handlers
    socketio_handlers.register_handlers()
    
    # Initialize and register external agent routes (if available)
    if AGENT_AVAILABLE and AgentRoutes is not None and AgentSocketIOHandlers is not None:
        try:
            agent_routes = AgentRoutes(auth_manager, tools_registry)
            agent_socketio_handlers = AgentSocketIOHandlers(socketio, auth_manager, tools_registry)
            agent_routes.register_routes(app)
            agent_socketio_handlers.register_handlers()
            logging.info("Agent routes registered successfully")
        except Exception as e:
            logging.error(f"Failed to register agent routes: {e}")
    else:
        logging.info("Agent features not available - server running in base mode")

    # Register document review routes if available
    if DOC_REVIEW_AVAILABLE and DocReviewRoutes is not None:
        try:
            doc_review_routes = DocReviewRoutes(auth_manager, tools_registry, mcp_handler, socketio)
            doc_review_routes.register_routes(app)
            logging.info("Document review routes registered successfully")
        except Exception as e:  # pylint: disable=broad-except
            logging.error(f"Failed to register doc review routes: {e}")
    else:
        logging.info("Document review features not available")

    # Register model documentation routes if available
    if MODEL_DOC_AVAILABLE and ModelDocRoutes is not None:
        try:
            model_doc_routes = ModelDocRoutes(auth_manager, tools_registry, mcp_handler, socketio)
            model_doc_routes.register_routes(app)
            logging.info("Model documentation routes registered successfully")
        except Exception as e:  # pylint: disable=broad-except
            logging.error(f"Failed to register model doc routes: {e}")
    else:
        logging.info("Model documentation features not available")

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

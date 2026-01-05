"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Routes Module for SAJHA MCP Server
"""

from external.routes.base_routes import BaseRoutes
from external.routes.auth_routes import AuthRoutes
from external.routes.dashboard_routes import DashboardRoutes
from external.routes.tools_routes import ToolsRoutes
from external.routes.admin_routes import AdminRoutes
from external.routes.monitoring_routes import MonitoringRoutes
from external.routes.api_routes import ApiRoutes
from external.routes.socketio_handlers import SocketIOHandlers

__all__ = [
    'BaseRoutes',
    'AuthRoutes',
    'DashboardRoutes',
    'ToolsRoutes',
    'AdminRoutes',
    'MonitoringRoutes',
    'ApiRoutes',
    'SocketIOHandlers'
]
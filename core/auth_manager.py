"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Authentication and Authorization Manager
"""

import json
import secrets
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

class AuthManager:
    """
    Manages authentication and authorization for the MCP server
    """
    
    def __init__(self, users_config_path: str = 'config/users.json'):
        """
        Initialize the AuthManager
        
        Args:
            users_config_path: Path to users configuration file
        """
        self.users_config_path = users_config_path
        self.users: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Load users configuration
        self.load_users()
        
        # Session configuration
        self.session_timeout_minutes = 60
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 5
    
    def load_users(self):
        """Load users from configuration file"""
        try:
            config_path = Path(self.users_config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.users = {user['user_id']: user for user in config.get('users', [])}
                    self.logger.info(f"Loaded {len(self.users)} users from configuration")
            else:
                # Create default admin user if no config exists
                default_config = {
                    "users": [
                        {
                            "user_id": "admin",
                            "user_name": "Administrator",
                            "password": "admin123",
                            "roles": ["admin"],
                            "tools": ["*"],
                            "domains": [],
                            "enabled": True,
                            "email": "admin@example.com",
                            "created_at": datetime.now().isoformat() + "Z"
                        }
                    ]
                }
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                self.users = {user['user_id']: user for user in default_config['users']}
                self.logger.info("Created default admin user configuration")
        except Exception as e:
            self.logger.error(f"Error loading users configuration: {e}")
            self.users = {}
    
    def authenticate(self, user_id: str, password: str) -> Optional[str]:
        """
        Authenticate a user and create a session
        
        Args:
            user_id: User identifier
            password: User password
            
        Returns:
            Session token if successful, None otherwise
        """
        with self._lock:
            # Check if user is locked out
            if self.is_user_locked_out(user_id):
                self.logger.warning(f"Login attempt for locked out user: {user_id}")
                return None
            
            # Check if user exists and is enabled
            user = self.users.get(user_id)
            if not user or not user.get('enabled', True):
                self.record_failed_attempt(user_id)
                self.logger.warning(f"Login attempt for invalid/disabled user: {user_id}")
                return None
            
            # Verify password
            if user.get('password') != password:
                self.record_failed_attempt(user_id)
                self.logger.warning(f"Invalid password for user: {user_id}")
                return None
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            session_data = {
                'user_id': user_id,
                'user_name': user.get('user_name', user_id),
                'roles': user.get('roles', []),
                'tools': user.get('tools', []),
                'domains': user.get('domains', []),  # NEW: Domain assignment
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            self.sessions[session_token] = session_data
            
            # Update last login
            self.update_last_login(user_id)

            # Clear failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]

            self.logger.info(f"User authenticated successfully: {user_id}")
            return session_token

    def validate_session(self, token: str) -> Optional[Dict]:
        """
        Validate a session token

        Args:
            token: Session token

        Returns:
            Session data if valid, None otherwise
        """
        with self._lock:
            session = self.sessions.get(token)
            if not session:
                return None

            # Check session timeout
            timeout = timedelta(minutes=self.session_timeout_minutes)
            if datetime.now() - session['last_activity'] > timeout:
                del self.sessions[token]
                self.logger.info(f"Session expired for user: {session['user_id']}")
                return None

            # Update last activity
            session['last_activity'] = datetime.now()
            return session

    def logout(self, token: str) -> bool:
        """
        Logout a user by invalidating their session

        Args:
            token: Session token

        Returns:
            True if successful
        """
        with self._lock:
            if token in self.sessions:
                user_id = self.sessions[token]['user_id']
                del self.sessions[token]
                self.logger.info(f"User logged out: {user_id}")
                return True
            return False

    def has_tool_access(self, session: Dict, tool_name: str) -> bool:
        """
        Check if a session has access to a specific tool

        Args:
            session: Session data
            tool_name: Name of the tool

        Returns:
            True if user has access to the tool
        """
        tools = session.get('tools', [])
        if '*' in tools or tool_name in tools:
            return True

        # Check if user has admin role
        if 'admin' in session.get('roles', []):
            return True

        return False

    def is_admin(self, session: Dict) -> bool:
        """
        Check if a session has admin privileges

        Args:
            session: Session data

        Returns:
            True if user is admin
        """
        return 'admin' in session.get('roles', [])
    
    def is_super_admin(self, session: Dict) -> bool:
        """
        Check if a session has super admin privileges.
        Super admin sees all workflows regardless of domain.

        Args:
            session: Session data

        Returns:
            True if user is super admin (has 'admin' role)
        """
        return self.is_admin(session)
    
    def is_domain_admin(self, session: Dict) -> bool:
        """
        Check if a session has domain admin privileges.
        Domain admin can create/edit workflows for their domains.

        Args:
            session: Session data

        Returns:
            True if user has 'domain_admin' role or is super admin
        """
        roles = session.get('roles', [])
        return 'domain_admin' in roles or 'admin' in roles
    
    def get_user_domains(self, user_id: str) -> List[str]:
        """
        Get list of domains assigned to a user.

        Args:
            user_id: User identifier

        Returns:
            List of domain names (empty list if user not found or no domains)
        """
        user = self.users.get(user_id)
        if not user:
            return []
        return user.get('domains', [])

    def get_user_accessible_tools(self, session: Dict) -> List[str]:
        """
        Get list of tools accessible to a user

        Args:
            session: Session data

        Returns:
            List of accessible tool names
        """
        tools = session.get('tools', [])
        if '*' in tools or self.is_admin(session):
            return ['*']  # All tools
        return tools

    def record_failed_attempt(self, user_id: str):
        """Record a failed login attempt"""
        with self._lock:
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = []

            self.failed_attempts[user_id].append(datetime.now())

            # Clean up old attempts
            cutoff = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt > cutoff
            ]

    def is_user_locked_out(self, user_id: str) -> bool:
        """Check if a user is locked out due to too many failed attempts"""
        with self._lock:
            if user_id not in self.failed_attempts:
                return False

            # Clean up old attempts
            cutoff = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
            recent_attempts = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt > cutoff
            ]

            return len(recent_attempts) >= self.max_login_attempts

    def save_users(self):
        """Save users configuration to file"""
        try:
            config = {"users": list(self.users.values())}
            config_path = Path(self.users_config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            self.logger.info(f"Saved {len(self.users)} users to configuration")
        except Exception as e:
            self.logger.error(f"Error saving users configuration: {e}")

    # =========================================================================
    # User Management Methods (for Admin UI)
    # =========================================================================

    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Get user by ID

        Args:
            user_id: User identifier

        Returns:
            User data dictionary or None
        """
        with self._lock:
            return self.users.get(user_id)

    def get_all_users(self) -> List[Dict]:
        """
        Get all users (includes passwords for admin editing)

        Returns:
            List of user dictionaries
        """
        with self._lock:
            return list(self.users.values())

    def create_user(self, user_data: Dict) -> bool:
        """
        Create a new user

        Args:
            user_data: User configuration dictionary

        Returns:
            True if successful
        """
        with self._lock:
            try:
                user_id = user_data.get('user_id')

                if not user_id:
                    self.logger.error("Cannot create user: user_id is required")
                    return False

                if user_id in self.users:
                    self.logger.error(f"Cannot create user: {user_id} already exists")
                    return False

                # Set creation timestamp if not provided
                if 'created_at' not in user_data:
                    user_data['created_at'] = datetime.now().isoformat() + 'Z'

                # Set defaults
                user_data.setdefault('enabled', True)
                user_data.setdefault('roles', ['user'])
                user_data.setdefault('tools', ['*'])

                # Add user to memory
                self.users[user_id] = user_data

                # Save to file
                self.save_users()

                self.logger.info(f"User created: {user_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error creating user: {e}")
                return False

    def update_user(self, user_id: str, user_data: Dict) -> bool:
        """
        Update existing user

        Args:
            user_id: User identifier
            user_data: Updated user configuration

        Returns:
            True if successful
        """
        with self._lock:
            try:
                if user_id not in self.users:
                    self.logger.error(f"User {user_id} not found")
                    return False

                # Keep original created_at if not provided
                if 'created_at' not in user_data and 'created_at' in self.users[user_id]:
                    user_data['created_at'] = self.users[user_id]['created_at']

                # If password is empty or None, keep the old password
                if not user_data.get('password'):
                    user_data['password'] = self.users[user_id].get('password', '')

                # Update user in memory (replace entire user object)
                self.users[user_id] = user_data

                # Save to file
                self.save_users()

                self.logger.info(f"User updated: {user_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error updating user: {e}")
                return False

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user

        Args:
            user_id: User identifier

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Prevent deleting admin
                if user_id == 'admin':
                    self.logger.error("Cannot delete admin user")
                    return False

                if user_id not in self.users:
                    self.logger.error(f"User {user_id} not found")
                    return False

                # Remove from memory
                del self.users[user_id]

                # Save to file
                self.save_users()

                # Invalidate all sessions for this user
                tokens_to_remove = [
                    token for token, session in self.sessions.items()
                    if session['user_id'] == user_id
                ]
                for token in tokens_to_remove:
                    del self.sessions[token]

                self.logger.info(f"User deleted: {user_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error deleting user: {e}")
                return False

    def enable_user(self, user_id: str) -> bool:
        """
        Enable a user account

        Args:
            user_id: User identifier

        Returns:
            True if successful
        """
        with self._lock:
            try:
                if user_id not in self.users:
                    self.logger.error(f"User {user_id} not found")
                    return False

                self.users[user_id]['enabled'] = True
                self.save_users()

                self.logger.info(f"User enabled: {user_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error enabling user: {e}")
                return False

    def disable_user(self, user_id: str) -> bool:
        """
        Disable a user account

        Args:
            user_id: User identifier

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Prevent disabling admin
                if user_id == 'admin':
                    self.logger.error("Cannot disable admin user")
                    return False

                if user_id not in self.users:
                    self.logger.error(f"User {user_id} not found")
                    return False

                self.users[user_id]['enabled'] = False
                self.save_users()

                # Invalidate all active sessions for this user
                tokens_to_remove = [
                    token for token, session in self.sessions.items()
                    if session['user_id'] == user_id
                ]
                for token in tokens_to_remove:
                    del self.sessions[token]

                self.logger.info(f"User disabled: {user_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error disabling user: {e}")
                return False

    def update_last_login(self, user_id: str):
        """
        Update user's last login timestamp

        Args:
            user_id: User identifier
        """
        try:
            if user_id in self.users:
                self.users[user_id]['last_login'] = datetime.now().isoformat() + 'Z'
                self.save_users()
        except Exception as e:
            self.logger.error(f"Error updating last login: {e}")

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================

    def add_user(self, user_data: Dict) -> bool:
        """
        Add a new user (legacy method, use create_user instead)

        Args:
            user_data: User data dictionary

        Returns:
            True if successful
        """
        return self.create_user(user_data)
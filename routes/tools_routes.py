"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Tools Routes for SAJHA MCP Server
"""

import logging
from flask import render_template
from routes.base_routes import BaseRoutes


class ToolsRoutes(BaseRoutes):
    """Tools-related routes"""

    def __init__(self, auth_manager, tools_registry):
        """Initialize tools routes"""
        super().__init__(auth_manager, tools_registry)

    def register_routes(self, app):
        """Register tools routes"""

        @app.route('/tools')
        @app.route('/tools/list')
        @self.login_required
        def tools_list():
            """Tools list page"""
            user_session = self.get_user_session()

            # Get all tools
            tools = self.tools_registry.get_all_tools()

            # Filter based on user permissions
            accessible_tools = self.auth_manager.get_user_accessible_tools(user_session)
            if '*' not in accessible_tools:
                tools = [t for t in tools if t['name'] in accessible_tools]

            return render_template('tools_list.html',
                                 user=user_session,
                                 tools=tools)

        @app.route('/tools/<tool_name>/schema')
        @self.login_required
        def tool_schema(tool_name):
            """Tool schema display page"""
            user_session = self.get_user_session()

            # Check if user has access to this tool
            if not self.auth_manager.has_tool_access(user_session, tool_name):
                return render_template('error.html',
                                     error="Access Denied",
                                     message=f"You don't have permission to view {tool_name}"), 403

            # Get tool details
            tool = self.tools_registry.get_tool(tool_name)
            if not tool:
                return render_template('error.html',
                                     error="Tool Not Found",
                                     message=f"Tool {tool_name} not found"), 404

            # Get tool data in MCP format
            tool_data = tool.to_mcp_format()

            # Import json for pretty printing
            import json
            schema_json = json.dumps(tool_data, indent=2)

            # Load tool configuration from config/tools folder
            config_json = None
            try:
                from pathlib import Path
                config_file = Path(self.tools_registry.tools_config_dir) / f"{tool_name}.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    config_json = json.dumps(config_data, indent=2)
            except Exception as e:
                config_json = f"Error loading configuration: {str(e)}"

            # Load literature content from config/literature folder
            literature_content = ""
            try:
                from pathlib import Path
                # Assume config/literature is at the same level as config/tools
                config_dir = Path(self.tools_registry.tools_config_dir).parent
                literature_file = config_dir / 'literature' / f"{tool_name}.txt"
                if literature_file.exists():
                    with open(literature_file, 'r', encoding='utf-8') as f:
                        literature_content = f.read()
            except Exception as e:
                literature_content = ""

            return render_template('tool_schema.html',
                                 user=user_session,
                                 tool=tool_data,
                                 schema_json=schema_json,
                                 config_json=config_json,
                                 literature_content=literature_content)

        @app.route('/agent/chat')
        @self.login_required
        def agent_chat():
            """Simple chat UI for parquet_agent showing response and prompt monitor."""
            from flask import make_response, request, redirect, url_for
            import time
            import hashlib
            user_session = self.get_user_session()
            # Ensure the agent tool exists; if not, page still renders but calls will fail gracefully
            cache_bust_value = int(time.time())
            
            # Read the actual template file to get its hash for cache busting
            try:
                import os
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

        @app.route('/api/agent/welcome-message')
        @self.login_required
        def get_welcome_message():
            """Get the welcome message for the agent chat interface"""
            import json
            from pathlib import Path
            from flask import jsonify, request
            
            try:
                # Get filename from query parameter, default to agent_welcome_message.json
                filename = request.args.get('file', 'agent_welcome_message.json')
                
                # Security: prevent path traversal
                if '..' in filename or '/' in filename or '\\' in filename:
                    return jsonify({"error": "Invalid filename"}), 400
                
                # Look in config/agent_welcome/ directory
                config_file = Path('config') / 'agent_welcome' / filename
                
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        return jsonify(json.load(f))
                else:
                    # Return a default message if file doesn't exist
                    return jsonify({
                        "title": "MR Limits AI assisted analysis",
                        "subtitle": "Ask questions about market risk limits data",
                        "sections": []
                    })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/agent/config-files', methods=['GET'])
        @self.login_required
        def list_config_files():
            """List JSON config files in specific directories based on type"""
            from pathlib import Path
            from flask import jsonify, request
            import os
            
            try:
                config_type = request.args.get('type', 'all')  # 'data_dict', 'agent_prompts', 'welcome', or 'all'
                config_dir = Path('config')
                files = []
                
                if config_type == 'agent_prompts':
                    # Only show files from config/agent/
                    agent_dir = config_dir / 'agent'
                    if agent_dir.exists():
                        for file_path in agent_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.json':
                                files.append({
                                    'name': file_path.name,
                                    'path': str(file_path.relative_to(Path('.'))),
                                    'size': file_path.stat().st_size,
                                    'modified': os.path.getmtime(file_path)
                                })
                elif config_type == 'data_dict':
                    # Only show files from config/data_dictionary/
                    data_dict_dir = config_dir / 'data_dictionary'
                    if data_dict_dir.exists():
                        for file_path in data_dict_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.json':
                                files.append({
                                    'name': file_path.name,
                                    'path': str(file_path.relative_to(Path('.'))),
                                    'size': file_path.stat().st_size,
                                    'modified': os.path.getmtime(file_path)
                                })
                elif config_type == 'welcome':
                    # Only show files from config/agent_welcome/
                    welcome_dir = config_dir / 'agent_welcome'
                    if welcome_dir.exists():
                        for file_path in welcome_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.json':
                                files.append({
                                    'name': file_path.name,
                                    'path': str(file_path.relative_to(Path('.'))),
                                    'size': file_path.stat().st_size,
                                    'modified': os.path.getmtime(file_path)
                                })
                else:
                    # 'all' - show files from both directories (for backward compatibility)
                    if config_dir.exists():
                        for file_path in config_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.json' and file_path.name != 'users.json':
                                files.append({
                                    'name': file_path.name,
                                    'path': str(file_path.relative_to(Path('.'))),
                                    'size': file_path.stat().st_size,
                                    'modified': os.path.getmtime(file_path)
                                })
                    
                    agent_dir = config_dir / 'agent'
                    if agent_dir.exists():
                        for file_path in agent_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.json':
                                files.append({
                                    'name': file_path.name,
                                    'path': str(file_path.relative_to(Path('.'))),
                                    'size': file_path.stat().st_size,
                                    'modified': os.path.getmtime(file_path)
                                })
                    
                    # Remove duplicates
                    seen = set()
                    unique_files = []
                    for f in files:
                        if f['name'] not in seen:
                            seen.add(f['name'])
                            unique_files.append(f)
                    files = unique_files
                
                # Sort by name
                files.sort(key=lambda x: x['name'])
                
                return jsonify({'files': files})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/agent/data-analysis-agents', methods=['GET', 'POST'])
        @self.login_required
        def data_analysis_agents():
            """List or create data analysis agent configurations"""
            from pathlib import Path
            from flask import jsonify, request
            import json
            import os
            from datetime import datetime
            
            agents_dir = Path('external') / 'config' / 'data_analysis_agents'
            agents_dir.mkdir(parents=True, exist_ok=True)
            
            if request.method == 'GET':
                # List all agents
                agents = []
                if agents_dir.exists():
                    for file_path in agents_dir.iterdir():
                        if file_path.is_file() and file_path.suffix == '.json':
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    agent_data = json.load(f)
                                    agent_data['id'] = file_path.stem  # Use filename without extension as ID
                                    agents.append(agent_data)
                            except Exception as e:
                                logging.error(f"Error reading agent file {file_path}: {e}")
                                continue
                
                # Sort by domain and name
                agents.sort(key=lambda x: (x.get('domain', ''), x.get('name', '')))
                
                return jsonify({'agents': agents})
            
            elif request.method == 'POST':
                # Create new agent
                try:
                    data = request.get_json()
                    if not data:
                        return jsonify({"error": "No data provided"}), 400
                    
                    domain = data.get('domain', '').strip()
                    name = data.get('name', '').strip()
                    data_dict_file = data.get('data_dict_file', '').strip()
                    agent_prompts_file = data.get('agent_prompts_file', '').strip()
                    welcome_tips_file = data.get('welcome_tips_file', '').strip()
                    
                    if not domain or not name:
                        return jsonify({"error": "Domain and Name are required"}), 400
                    
                    if not data_dict_file or not agent_prompts_file or not welcome_tips_file:
                        return jsonify({"error": "All three config files are required"}), 400
                    
                    # Create agent ID from domain and name (sanitized)
                    agent_id = f"{domain}_{name}".lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
                    # Remove special characters
                    import re
                    agent_id = re.sub(r'[^a-z0-9_]', '', agent_id)
                    
                    # Check if agent with same domain+name already exists
                    existing_file = agents_dir / f"{agent_id}.json"
                    if existing_file.exists():
                        return jsonify({"error": f"Agent with domain '{domain}' and name '{name}' already exists"}), 400
                    
                    # Create agent config
                    agent_config = {
                        'domain': domain,
                        'name': name,
                        'data_dict_file': data_dict_file,
                        'agent_prompts_file': agent_prompts_file,
                        'welcome_tips_file': welcome_tips_file,
                        'created_at': data.get('created_at', datetime.utcnow().isoformat() + 'Z')
                    }
                    
                    # Save to file
                    agent_file = agents_dir / f"{agent_id}.json"
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_config, f, indent=2, ensure_ascii=False)
                    
                    return jsonify({
                        "success": True,
                        "agent_id": agent_id,
                        "message": f"Agent '{domain} - {name}' saved successfully"
                    })
                    
                except Exception as e:
                    logging.error(f"Error saving agent: {e}", exc_info=True)
                    return jsonify({"error": str(e)}), 500
        
        @app.route('/api/agent/data-analysis-agents/<agent_id>', methods=['GET'])
        @self.login_required
        def get_data_analysis_agent(agent_id):
            """Get a specific agent configuration by ID"""
            from pathlib import Path
            from flask import jsonify
            import json
            
            agents_dir = Path('external') / 'config' / 'data_analysis_agents'
            agent_file = agents_dir / f"{agent_id}.json"
            
            if not agent_file.exists():
                return jsonify({"error": "Agent not found"}), 404
            
            try:
                with open(agent_file, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)
                    agent_data['id'] = agent_id
                    return jsonify({'agent': agent_data})
            except Exception as e:
                logging.error(f"Error reading agent file {agent_file}: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/agent/config-upload', methods=['POST'])
        @self.login_required
        def upload_config_file():
            """Upload a config file to the config directory"""
            from pathlib import Path
            from flask import jsonify, request
            import os
            from werkzeug.utils import secure_filename
            
            try:
                if 'file' not in request.files:
                    return jsonify({"error": "No file provided"}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                # Get config type to determine save location
                config_type = request.form.get('config_type', 'general')  # 'data_dict', 'agent_prompts', 'welcome', or 'general'
                
                # Validate filename
                filename = secure_filename(file.filename)
                if not filename.endswith('.json'):
                    return jsonify({"error": "Only JSON files are allowed"}), 400
                
                # Security: prevent path traversal
                if '..' in filename or '/' in filename or '\\' in filename:
                    return jsonify({"error": "Invalid filename"}), 400
                
                # Determine save location based on config type
                config_dir = Path('config')
                if config_type == 'agent_prompts':
                    # Save agent prompts to config/agent/ subdirectory
                    agent_dir = config_dir / 'agent'
                    agent_dir.mkdir(exist_ok=True)
                    file_path = agent_dir / filename
                elif config_type == 'data_dict':
                    # Save to config/data_dictionary/
                    data_dict_dir = config_dir / 'data_dictionary'
                    data_dict_dir.mkdir(exist_ok=True)
                    file_path = data_dict_dir / filename
                elif config_type == 'welcome':
                    # Save to config/agent_welcome/
                    welcome_dir = config_dir / 'agent_welcome'
                    welcome_dir.mkdir(exist_ok=True)
                    file_path = welcome_dir / filename
                else:
                    # Default: save to config root
                    config_dir.mkdir(exist_ok=True)
                    file_path = config_dir / filename
                
                file.save(str(file_path))
                
                return jsonify({
                    "success": True,
                    "filename": filename,
                    "path": str(file_path.relative_to(Path('.'))),
                    "message": f"File {filename} uploaded successfully"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route('/tools/<tool_name>/literature/save', methods=['POST'])
        @self.login_required
        def save_tool_literature(tool_name):
            """Save literature content for a tool"""
            user_session = self.get_user_session()

            # Check if user has access to this tool
            if not self.auth_manager.has_tool_access(user_session, tool_name):
                from flask import jsonify
                return jsonify({
                    'success': False,
                    'error': 'Access denied'
                }), 403

            # Get tool details to verify it exists
            tool = self.tools_registry.get_tool(tool_name)
            if not tool:
                from flask import jsonify
                return jsonify({
                    'success': False,
                    'error': 'Tool not found'
                }), 404

            try:
                from flask import request
                from pathlib import Path
                import os

                # Get the literature content from request
                literature_content = request.form.get('literature_content', '')

                # Determine literature file path
                config_dir = Path(self.tools_registry.tools_config_dir).parent
                literature_dir = config_dir / 'literature'

                # Create literature directory if it doesn't exist
                literature_dir.mkdir(parents=True, exist_ok=True)

                literature_file = literature_dir / f"{tool_name}.txt"

                # Write the content to file
                with open(literature_file, 'w', encoding='utf-8') as f:
                    f.write(literature_content)

                from flask import jsonify
                return jsonify({
                    'success': True,
                    'message': 'Literature content saved successfully'
                })

            except Exception as e:
                from flask import jsonify
                return jsonify({
                    'success': False,
                    'error': f'Error saving literature: {str(e)}'
                }), 500

        @app.route('/tooljson/list')
        @self.login_required
        def tools_list_json():
            """Get list of all tools in JSON format"""
            user_session = self.get_user_session()

            # Get all tools
            tools = self.tools_registry.get_all_tools()

            # Filter based on user permissions
            accessible_tools = self.auth_manager.get_user_accessible_tools(user_session)
            if '*' not in accessible_tools:
                tools = [t for t in tools if t['name'] in accessible_tools]

            from flask import jsonify
            return jsonify({
                'success': True,
                'count': len(tools),
                'tools': tools
            })

        @app.route('/tooljson/<tool_name>')
        @self.login_required
        def tool_json(tool_name):
            """Get tool schema and config in JSON format"""
            user_session = self.get_user_session()

            # Check if user has access to this tool
            if not self.auth_manager.has_tool_access(user_session, tool_name):
                from flask import jsonify
                return jsonify({
                    'success': False,
                    'error': 'Access denied'
                }), 403

            # Get tool details
            tool = self.tools_registry.get_tool(tool_name)
            if not tool:
                from flask import jsonify
                return jsonify({
                    'success': False,
                    'error': 'Tool not found'
                }), 404

            # Get tool schema
            tool_schema = tool.to_mcp_format()

            # Load tool configuration from config/tools folder
            tool_config = None
            try:
                import json
                from pathlib import Path
                config_file = Path(self.tools_registry.tools_config_dir) / f"{tool_name}.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        tool_config = json.load(f)
            except Exception as e:
                tool_config = {'error': f'Error loading configuration: {str(e)}'}

            # Load literature content from config/literature folder
            literature_content = ""
            try:
                from pathlib import Path
                # Assume config/literature is at the same level as config/tools
                config_dir = Path(self.tools_registry.tools_config_dir).parent
                literature_file = config_dir / 'literature' / f"{tool_name}.txt"
                if literature_file.exists():
                    with open(literature_file, 'r', encoding='utf-8') as f:
                        literature_content = f.read()
            except Exception as e:
                # If there's an error reading literature, just leave it empty
                literature_content = ""

            from flask import jsonify
            return jsonify({
                'success': True,
                'tool_name': tool_name,
                'schema': tool_schema,
                'config': tool_config,
                'literature': literature_content
            })

        @app.route('/tooljsonall')
        @self.login_required
        def tools_all_json_list():
            """Get all tools with their schemas and configs in JSON array format"""
            user_session = self.get_user_session()

            # Get all tools
            all_tools = self.tools_registry.get_all_tools()

            # Filter based on user permissions
            accessible_tools = self.auth_manager.get_user_accessible_tools(user_session)
            if '*' not in accessible_tools:
                all_tools = [t for t in all_tools if t['name'] in accessible_tools]

            # Build the comprehensive tools list
            tools_json_list = []

            import json
            from pathlib import Path

            for tool_info in all_tools:
                tool_name = tool_info['name']

                # Get tool instance
                tool = self.tools_registry.get_tool(tool_name)
                if not tool:
                    continue

                # Get tool schema
                tool_schema = tool.to_mcp_format()

                # Load tool configuration from config/tools folder
                tool_config = None
                try:
                    config_file = Path(self.tools_registry.tools_config_dir) / f"{tool_name}.json"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            tool_config = json.load(f)
                except Exception as e:
                    tool_config = {'error': f'Error loading configuration: {str(e)}'}

                # Load literature content from config/literature folder
                literature_content = ""
                try:
                    # Assume config/literature is at the same level as config/tools
                    config_dir = Path(self.tools_registry.tools_config_dir).parent
                    literature_file = config_dir / 'literature' / f"{tool_name}.txt"
                    if literature_file.exists():
                        with open(literature_file, 'r', encoding='utf-8') as f:
                            literature_content = f.read()
                except Exception as e:
                    # If there's an error reading literature, just leave it empty
                    literature_content = ""

                # Add to list
                tools_json_list.append({
                    tool_name: {
                        'schema': tool_schema,
                        'config': tool_config,
                        'literature': literature_content
                    }
                })

            from flask import jsonify
            return jsonify({
                'success': True,
                'count': len(tools_json_list),
                'tools': tools_json_list
            })

        @app.route('/tools/execute/<tool_name>')
        @self.login_required
        def tool_execute(tool_name):
            """Tool execution page"""
            user_session = self.get_user_session()

            # Check if user has access to this tool
            if not self.auth_manager.has_tool_access(user_session, tool_name):
                return render_template('error.html',
                                     error="Access Denied",
                                     message=f"You don't have permission to use {tool_name}"), 403

            # Get tool details
            tool = self.tools_registry.get_tool(tool_name)
            if not tool:
                return render_template('error.html',
                                     error="Tool Not Found",
                                     message=f"Tool {tool_name} not found"), 404

            return render_template('tool_execute.html',
                                 user=user_session,
                                 tool=tool.to_mcp_format())
"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Admin Routes for SAJHA MCP Server
"""

from flask import render_template, jsonify, request
from pathlib import Path
import logging
from routes.base_routes import BaseRoutes
from werkzeug.utils import secure_filename

from external.agent.agent_registry import (
    validate_agent_type,
    validate_safe_name,
    slugify_name,
)
from core.db.session import get_db_session
from external.agent.persistence import (
    list_prompts,
    get_prompt_content,
    upsert_prompt,
    list_agents_db,
    get_agent_db,
    create_agent_db,
    update_agent_db,
    delete_agent_db,
    list_agent_prompts,
    get_agent_prompt,
    upsert_agent_prompt,
    delete_agent_prompt,
)

logger = logging.getLogger(__name__)


class AdminRoutes(BaseRoutes):
    """Admin-related routes for managing tools and users"""

    def __init__(self, auth_manager, tools_registry):
        """Initialize admin routes"""
        super().__init__(auth_manager, tools_registry)

    def register_routes(self, app):
        """Register admin routes"""

        @app.route('/admin')
        @self.admin_required
        def admin_panel():
            """Admin panel main page - supports section parameter"""
            user_session = self.get_user_session()
            section = request.args.get('section', 'chat-agents')  # Default to chat-agents
            return render_template('admin.html', user=user_session, section=section)

        @app.route('/api/admin/prompts')
        @self.admin_required
        def list_prompts_endpoint():
            """List all available prompts, optionally filtered by category"""
            try:
                category = request.args.get('category', '').strip() or None
                with get_db_session() as db:
                    prompts = list_prompts(db, category=category)
                    return jsonify({"prompts": prompts})
            except Exception as e:
                logger.error(f"Error listing prompts: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/prompts/<prompt_name>', methods=['GET'])
        @self.admin_required
        def get_prompt(prompt_name):
            """Get specific prompt content"""
            try:
                # Validate prompt name (security: prevent path traversal)
                if '..' in prompt_name or '/' in prompt_name or '\\' in prompt_name:
                    return jsonify({"error": "Invalid prompt name"}), 400
                
                with get_db_session() as db:
                    content = get_prompt_content(db, prompt_name)
                    if content is None:
                        return jsonify({"error": f"Prompt '{prompt_name}' not found"}), 404
                    return jsonify({
                        "name": prompt_name,
                        "content": content,
                        "filename": f"{prompt_name}.md"
                    })
            except Exception as e:
                logger.error(f"Error reading prompt {prompt_name}: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/prompts/<prompt_name>', methods=['POST'])
        @self.admin_required
        def save_prompt(prompt_name):
            """Save updated prompt content"""
            try:
                data = request.get_json()
                if not data or 'content' not in data:
                    return jsonify({"error": "Missing 'content' in request body"}), 400
                
                # Validate prompt name (security: prevent path traversal)
                if '..' in prompt_name or '/' in prompt_name or '\\' in prompt_name:
                    return jsonify({"error": "Invalid prompt name"}), 400
                
                category = data.get('category', 'structured')
                user_id = self.get_user_session().get('user_id')
                
                with get_db_session() as db:
                    upsert_prompt(db, prompt_name, category, data['content'], editor_user_id=user_id)
                    db.commit()
                    logger.info(f"Prompt '{prompt_name}' updated by user {user_id}")
                
                return jsonify({
                    "success": True,
                    "message": f"Prompt '{prompt_name}' saved successfully"
                })
            except Exception as e:
                logger.error(f"Error saving prompt {prompt_name}: {e}")
                return jsonify({"error": str(e)}), 500

        # ==================== Agent Management ====================
        @app.route('/api/admin/agents', methods=['GET'])
        @self.login_required  # Changed from admin_required - users need to see agents to use them
        def api_list_agents():
            try:
                with get_db_session() as db:
                    agents = list_agents_db(db)
                    return jsonify({"agents": agents})
            except Exception as e:
                logger.error(f"Error listing agents: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents/<agent_id>', methods=['GET'])
        @self.admin_required
        def api_get_agent(agent_id):
            try:
                with get_db_session() as db:
                    cfg = get_agent_db(db, agent_id)
                    if not cfg:
                        return jsonify({"error": "Agent not found"}), 404
                    return jsonify({"agent": cfg})
            except Exception as e:
                logger.error(f"Error getting agent {agent_id}: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents/<agent_id>', methods=['PUT'])
        @self.admin_required
        def api_update_agent(agent_id):
            """Update an existing agent (supports name, description, domain_file, data_folder, and data_files)"""
            MAX_FILE_BYTES = 10 * 1024 * 1024
            try:
                # Check if this is multipart/form-data (file upload) or JSON
                if request.content_type and 'multipart/form-data' in request.content_type:
                    # Handle file uploads (similar to create)
                    name = (request.form.get("name") or "").strip()
                    description = (request.form.get("description") or "").strip()
                    
                    if not name:
                        return jsonify({"error": "Agent name is required"}), 400
                    
                    domain_file = request.files.get("domain_file")
                    domain_filename = None
                    domain_text = None
                    if domain_file and domain_file.filename:
                        domain_filename = secure_filename(domain_file.filename)
                        if not (domain_filename.lower().endswith(".md") or domain_filename.lower().endswith(".txt")):
                            return jsonify({"error": "domain_file must be .md or .txt"}), 400
                        domain_bytes = domain_file.read()
                        if len(domain_bytes) > MAX_FILE_BYTES:
                            return jsonify({"error": "domain_file exceeds 10MB limit"}), 400
                        domain_text = domain_bytes.decode("utf-8", errors="replace")
                        domain_filename = domain_filename
                    
                    data_folder = None
                    data_folder_mode = request.form.get("data_folder_mode", "select")
                    if data_folder_mode == "select":
                        data_folder = (request.form.get("data_folder") or "").strip()
                    elif data_folder_mode == "create":
                        data_folder_name = (request.form.get("data_folder") or "").strip()
                        if not data_folder_name:
                            return jsonify({"error": "data_folder name is required when creating new folder"}), 400
                        # Slugify folder name for create mode
                        data_folder = slugify_name(data_folder_name)
                        validate_safe_name(data_folder, "data_folder")
                    # If data_folder_mode not provided, check if data_folder was directly provided
                    if not data_folder:
                        data_folder = (request.form.get("data_folder") or "").strip()
                    
                    # Store domain file content in DB (not on disk)
                    if domain_filename and domain_text:
                        # Content will be stored in domain_content column
                        pass
                    elif domain_file is None:
                        # Not updating domain file - get current values from DB
                        with get_db_session() as db:
                            current_agent = get_agent_db(db, agent_id)
                            if current_agent:
                                domain_filename = current_agent.get("domain_file")
                                domain_text = current_agent.get("domain_content")
                    
                    # Handle data files if provided
                    data_files = request.files.getlist("data_files")
                    if data_files and len(data_files) > 0:
                        if not data_folder:
                            # Use current data folder
                            with get_db_session() as db:
                                current_agent = get_agent_db(db, agent_id)
                                if current_agent:
                                    data_folder = current_agent.get("data_folder")
                                else:
                                    return jsonify({"error": "Agent not found"}), 404
                        
                        if not data_folder:
                            return jsonify({"error": "data_folder is required when uploading data files"}), 400
                        
                        base_folder = Path("external/datawarehouse") / data_folder
                        if data_folder_mode == "create":
                            base_folder.mkdir(parents=True, exist_ok=True)
                        elif not base_folder.exists():
                            return jsonify({"error": f"Data folder '{data_folder}' does not exist"}), 400
                        
                        for data_file in data_files:
                            if not data_file.filename:
                                continue
                            safe_name = secure_filename(data_file.filename)
                            if not safe_name:
                                continue
                            if not (safe_name.lower().endswith(".csv") or safe_name.lower().endswith(".parquet")):
                                return jsonify({"error": f"File '{safe_name}' must be .csv or .parquet"}), 400
                            data_bytes = data_file.read()
                            if len(data_bytes) > MAX_FILE_BYTES:
                                return jsonify({"error": f"File '{safe_name}' exceeds 10MB limit"}), 400
                            target_path = base_folder / safe_name
                            target_path.write_bytes(data_bytes)
                    
                    # Get model from form (optional update)
                    model = (request.form.get("model") or "").strip()
                    
                    # Update in DB
                    with get_db_session() as db:
                        update_data = {"name": name}
                        if description:
                            update_data["description"] = description
                        if domain_filename:
                            update_data["domain_file"] = domain_filename
                        if domain_text is not None:
                            update_data["domain_content"] = domain_text
                        if data_folder:
                            update_data["data_folder"] = data_folder
                        if model:
                            update_data["model"] = model
                        
                        agent = update_agent_db(
                            db,
                            agent_id=agent_id,
                            **update_data
                        )
                        if not agent:
                            return jsonify({"error": "Agent not found"}), 404
                        db.commit()
                        return jsonify({
                            "success": True,
                            "message": f"Agent '{name}' updated successfully",
                            "agent": get_agent_db(db, agent_id)
                        })
                else:
                    # JSON update (name/description/model only)
                    data = request.get_json()
                    if not data:
                        return jsonify({"error": "No data provided"}), 400
                    
                    with get_db_session() as db:
                        agent = update_agent_db(
                            db,
                            agent_id=agent_id,
                            name=data.get("name"),
                            description=data.get("description"),
                            model=data.get("model"),
                        )
                        if not agent:
                            return jsonify({"error": "Agent not found"}), 404
                        db.commit()
                        return jsonify({
                            "success": True,
                            "message": f"Agent '{agent_id}' updated successfully",
                            "agent": get_agent_db(db, agent_id)
                        })
            except Exception as e:
                logger.error(f"Error updating agent {agent_id}: {e}")
                return jsonify({"error": str(e)}), 500

        # ==================== Agent Prompt Management ====================
        @app.route('/api/admin/agents/<agent_id>/prompts', methods=['GET'])
        @self.admin_required
        def api_list_agent_prompts(agent_id):
            """List all prompts for an agent, showing which are overridden."""
            try:
                with get_db_session() as db:
                    prompts = list_agent_prompts(db, agent_id)
                    return jsonify({"prompts": prompts})
            except Exception as e:
                logger.error(f"Error listing agent prompts: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents/<agent_id>/prompts/<prompt_name>', methods=['GET'])
        @self.admin_required
        def api_get_agent_prompt(agent_id, prompt_name):
            """Get agent-specific prompt override."""
            try:
                with get_db_session() as db:
                    agent_prompt = get_agent_prompt(db, agent_id, prompt_name)
                    if agent_prompt:
                        return jsonify({
                            "name": agent_prompt.prompt_name,
                            "content": agent_prompt.content,
                            "is_active": agent_prompt.is_active
                        })
                    else:
                        # Return default prompt
                        agent = get_agent_db(db, agent_id)
                        category = "web_search" if agent and agent.get("agent_type") == "external" else "structured"
                        default_content = get_prompt_content(db, prompt_name, category=category)
                        return jsonify({
                            "name": prompt_name,
                            "content": default_content or "",
                            "is_active": False,
                            "is_default": True
                        })
            except Exception as e:
                logger.error(f"Error getting agent prompt: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents/<agent_id>/prompts/<prompt_name>', methods=['POST'])
        @self.admin_required
        def api_save_agent_prompt(agent_id, prompt_name):
            """Save agent-specific prompt override."""
            try:
                data = request.get_json()
                if not data or 'content' not in data:
                    return jsonify({"error": "Missing 'content' in request body"}), 400
                
                with get_db_session() as db:
                    upsert_agent_prompt(db, agent_id, prompt_name, data['content'], is_active=data.get('is_active', True))
                    db.commit()
                    return jsonify({
                        "success": True,
                        "message": f"Agent prompt '{prompt_name}' saved successfully"
                    })
            except Exception as e:
                logger.error(f"Error saving agent prompt: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents/<agent_id>/prompts/<prompt_name>', methods=['DELETE'])
        @self.admin_required
        def api_delete_agent_prompt(agent_id, prompt_name):
            """Delete agent-specific prompt override (revert to default)."""
            try:
                with get_db_session() as db:
                    delete_agent_prompt(db, agent_id, prompt_name)
                    db.commit()
                    return jsonify({
                        "success": True,
                        "message": f"Agent prompt override deleted, using default"
                    })
            except Exception as e:
                logger.error(f"Error deleting agent prompt: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents/<agent_id>', methods=['DELETE'])
        @self.admin_required
        def api_delete_agent(agent_id):
            try:
                with get_db_session() as db:
                    deleted = delete_agent_db(db, agent_id)
                    if not deleted:
                        return jsonify({"error": "Agent not found"}), 404
                    db.commit()
                    return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error deleting agent {agent_id}: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/datawarehouse/folders', methods=['GET'])
        @self.admin_required
        def api_list_datawarehouse_folders():
            try:
                base = Path("external/datawarehouse")
                base.mkdir(parents=True, exist_ok=True)
                folders = []
                for p in base.iterdir():
                    if p.is_dir():
                        folders.append(p.name)
                return jsonify({"folders": sorted(folders)})
            except Exception as e:
                logger.error(f"Error listing datawarehouse folders: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/datawarehouse/folders/<folder>/files', methods=['GET'])
        @self.admin_required
        def api_list_datawarehouse_files(folder):
            try:
                validate_safe_name(folder, "data_folder")
                base = Path("external/datawarehouse") / folder
                if not base.exists() or not base.is_dir():
                    return jsonify({"error": "Folder not found"}), 404
                files = []
                for p in base.iterdir():
                    if p.is_file():
                        files.append({"name": p.name, "size_bytes": p.stat().st_size})
                return jsonify({"files": sorted(files, key=lambda x: x["name"])})
            except Exception as e:
                logger.error(f"Error listing files for folder {folder}: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/datawarehouse/folders/<folder>/files/<filename>', methods=['DELETE'])
        @self.admin_required
        def api_delete_datawarehouse_file(folder, filename):
            try:
                validate_safe_name(folder, "data_folder")
                safe = secure_filename(filename or "")
                if not safe:
                    return jsonify({"error": "Invalid filename"}), 400
                target = Path("external/datawarehouse") / folder / safe
                if not target.exists() or not target.is_file():
                    return jsonify({"error": "File not found"}), 404
                target.unlink()
                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error deleting file {folder}/{filename}: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/admin/agents', methods=['POST'])
        @self.admin_required
        def api_create_agent():
            """
            Create a new agent instance (structured only for v1) and upload:
            - domain file (.md/.txt)
            - data files (.csv/.parquet), multiple, 10MB max per file
            """
            MAX_FILE_BYTES = 10 * 1024 * 1024
            try:
                name = (request.form.get("name") or "").strip()
                description = (request.form.get("description") or "").strip()
                agent_type = (request.form.get("agent_type") or "").strip()
                data_folder_mode = (request.form.get("data_folder_mode") or "").strip()
                data_folder_name = (request.form.get("data_folder_name") or "").strip()
                data_folder_existing = (request.form.get("data_folder_existing") or "").strip()

                validate_agent_type(agent_type)
                
                # Handle structured vs external agents
                data_folder = None
                if agent_type == "structured":
                    if data_folder_mode not in ("create", "select"):
                        return jsonify({"error": "data_folder_mode must be 'create' or 'select' for structured agents"}), 400
                    data_folder = data_folder_name if data_folder_mode == "create" else data_folder_existing
                    # Sanitize folder name: convert spaces to underscores, remove invalid chars
                    if data_folder_mode == "create":
                        data_folder = slugify_name(data_folder)
                    validate_safe_name(data_folder, "data_folder")
                elif agent_type == "external":
                    # External agents don't need data_folder
                    pass
                else:
                    return jsonify({"error": f"Agent type '{agent_type}' not yet supported"}), 400

                # Domain file required
                domain_file = request.files.get("domain_file")
                if not domain_file or not getattr(domain_file, "filename", ""):
                    return jsonify({"error": "domain_file is required (.md or .txt)"}), 400
                domain_filename = secure_filename(domain_file.filename)
                if not (domain_filename.lower().endswith(".md") or domain_filename.lower().endswith(".txt")):
                    return jsonify({"error": "domain_file must be .md or .txt"}), 400
                domain_bytes = domain_file.read()
                if len(domain_bytes) > MAX_FILE_BYTES:
                    return jsonify({"error": "domain_file exceeds 10MB limit"}), 400
                domain_text = domain_bytes.decode("utf-8", errors="replace")

                # If selecting existing folder, ensure it exists (structured agents only)
                if agent_type == "structured" and data_folder_mode == "select":
                    base_folder = Path("external/datawarehouse") / data_folder
                    if not base_folder.exists() or not base_folder.is_dir():
                        return jsonify({"error": f"Selected folder does not exist: {data_folder}"}), 400

                # Generate agent_id (use slugified name)
                from external.agent.agent_registry import slugify_name
                agent_id = slugify_name(name)
                
                # Get model from form (default to Sonnet for thinking support)
                model = (request.form.get("model") or "").strip() or "claude-3-sonnet-20240229"
                
                # Get web search specific fields (for external agents)
                tavily_api_key = (request.form.get("tavily_api_key") or "").strip()
                allowed_domains_str = (request.form.get("allowed_domains") or "").strip()
                blocked_domains_str = (request.form.get("blocked_domains") or "").strip()
                default_research_depth = (request.form.get("default_research_depth") or "").strip() or "standard"
                
                # Parse domain lists
                import json
                allowed_domains = [d.strip() for d in allowed_domains_str.split(",") if d.strip()] if allowed_domains_str else None
                blocked_domains = [d.strip() for d in blocked_domains_str.split(",") if d.strip()] if blocked_domains_str else None
                
                # Store domain file content in DB (not on disk)
                # Only store filename for reference, content goes in domain_content column
                
                # Write to DB only (no file-based storage)
                with get_db_session() as db:
                    # Check if agent already exists
                    existing = get_agent_db(db, agent_id)
                    if existing:
                        return jsonify({"error": f"Agent with ID '{agent_id}' already exists. Please use a different name or edit the existing agent."}), 409
                    
                    create_agent_db(
                        db,
                        agent_id=agent_id,
                        name=name,
                        agent_type=agent_type,
                        description=description,
                        domain_file=domain_filename,  # Filename for reference
                        domain_content=domain_text,  # Content stored in DB
                        data_folder=data_folder,
                        model=model,
                        tavily_api_key=tavily_api_key if tavily_api_key else None,
                        search_scope_allowed_domains=allowed_domains,
                        search_scope_blocked_domains=blocked_domains,
                        default_research_depth=default_research_depth if agent_type == "external" else None,
                    )
                    db.commit()
                    # Get the created agent to return it
                    created_agent = get_agent_db(db, agent_id)

                # Upload data files (structured agents only)
                uploaded = []
                rejected = []
                if agent_type == "structured":
                    files = request.files.getlist("data_files") or []
                    base_folder = Path("external/datawarehouse") / data_folder
                    base_folder.mkdir(parents=True, exist_ok=True)

                    for f in files:
                        if not f or not getattr(f, "filename", ""):
                            continue
                        fname = secure_filename(f.filename)
                        lower = fname.lower()
                        if not (lower.endswith(".csv") or lower.endswith(".parquet")):
                            rejected.append({"name": fname, "reason": "Only .csv and .parquet supported"})
                            continue
                        b = f.read()
                        if len(b) > MAX_FILE_BYTES:
                            rejected.append({"name": fname, "reason": "File exceeds 10MB limit"})
                            continue

                        target = base_folder / fname
                        # Avoid overwrite by suffixing
                        if target.exists():
                            stem = target.stem
                            suf = target.suffix
                            i = 2
                            while True:
                                candidate = base_folder / f"{stem}-{i}{suf}"
                                if not candidate.exists():
                                    target = candidate
                                    break
                                i += 1
                        target.write_bytes(b)
                        uploaded.append({"name": target.name, "size_bytes": len(b)})

                return jsonify({
                    "success": True,
                    "message": f"Agent '{name}' created successfully",
                    "agent": created_agent,
                    "uploaded_files": uploaded,
                    "rejected_files": rejected,
                })
            except Exception as e:
                logger.error(f"Error creating agent: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/admin/tools')
        @self.admin_required
        def admin_tools():
            """Admin tools management page"""
            user_session = self.get_user_session()

            # Get all tools with metrics
            tools_metrics = self.tools_registry.get_tool_metrics()
            tool_errors = self.tools_registry.get_tool_errors()

            return render_template('admin_tools.html',
                                 user=user_session,
                                 tools_metrics=tools_metrics,
                                 tool_errors=tool_errors)

        @app.route('/admin/tools/<tool_name>/config')
        @self.admin_required
        def tool_config_page(tool_name):
            """Tool configuration editor page"""
            user_session = self.get_user_session()

            # Verify tool exists
            tool = self.tools_registry.get_tool(tool_name)
            if not tool and tool_name not in self.tools_registry.tool_configs:
                return render_template('error.html',
                                     user=user_session,
                                     error="Tool Not Found",
                                     message=f"Tool '{tool_name}' does not exist"), 404

            return render_template('tool_config.html',
                                 user=user_session,
                                 tool_name=tool_name)

        @app.route('/admin/users')
        @self.admin_required
        def admin_users():
            """Admin users management page"""
            user_session = self.get_user_session()

            # Get all users
            users = self.auth_manager.get_all_users()

            return render_template('admin_users.html',
                                 user=user_session,
                                 users=users)

        @app.route('/admin/users/<user_id>/config')
        @self.admin_required
        def user_config_page(user_id):
            """User configuration editor page"""
            user_session = self.get_user_session()

            # Verify user exists
            user_data = self.auth_manager.get_user(user_id)
            if not user_data:
                return render_template('error.html',
                                     user=user_session,
                                     error="User Not Found",
                                     message=f"User '{user_id}' does not exist"), 404

            return render_template('user_config.html',
                                 user=user_session,
                                 user_id=user_id)
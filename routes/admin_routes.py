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
    create_agent,
    delete_agent,
    get_agent,
    list_agents,
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
    delete_agent_db,
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
            """Admin panel main page"""
            user_session = self.get_user_session()
            return render_template('admin.html', user=user_session)

        @app.route('/api/admin/prompts')
        @self.admin_required
        def list_prompts():
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
        @self.admin_required
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

        @app.route('/api/admin/agents/<agent_id>', methods=['DELETE'])
        @self.admin_required
        def api_delete_agent(agent_id):
            try:
                with get_db_session() as db:
                    deleted = delete_agent_db(db, agent_id)
                    if not deleted:
                        return jsonify({"error": "Agent not found"}), 404
                    db.commit()
                    # Also delete from file-based registry for now (dual-write)
                    try:
                        delete_agent(agent_id)
                    except Exception:
                        pass  # Ignore file deletion errors
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
                if agent_type != "structured":
                    return jsonify({"error": "Coming soon: only structured agents are supported in v1"}), 400

                if data_folder_mode not in ("create", "select"):
                    return jsonify({"error": "data_folder_mode must be 'create' or 'select'"}), 400

                data_folder = data_folder_name if data_folder_mode == "create" else data_folder_existing
                # Sanitize folder name: convert spaces to underscores, remove invalid chars
                # This allows user-friendly names like "Financial Analysis" -> "financial_analysis"
                if data_folder_mode == "create":
                    data_folder = slugify_name(data_folder)
                validate_safe_name(data_folder, "data_folder")

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

                # If selecting existing folder, ensure it exists
                base_folder = Path("external/datawarehouse") / data_folder
                if data_folder_mode == "select":
                    if not base_folder.exists() or not base_folder.is_dir():
                        return jsonify({"error": f"Selected folder does not exist: {data_folder}"}), 400

                # Create agent config + domain file + ensure folder exists
                # Dual-write: file-based (legacy) + DB
                cfg = create_agent(
                    name=name,
                    description=description,
                    agent_type=agent_type,
                    domain_text=domain_text,
                    data_folder=data_folder,
                )
                
                # Also write to DB
                with get_db_session() as db:
                    create_agent_db(
                        db,
                        agent_id=cfg['id'],
                        name=cfg['name'],
                        agent_type=cfg['agent_type'],
                        description=cfg.get('description'),
                        domain_file=cfg.get('domain_file'),
                        data_folder=cfg.get('data_folder'),
                    )
                    db.commit()

                # Upload data files
                uploaded = []
                rejected = []
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
                    "agent": cfg,
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
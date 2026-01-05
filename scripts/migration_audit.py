#!/usr/bin/env python3
"""
Migration Audit Script
Compares current project with base SAJHA MCP Server to identify custom code.
"""
import os
import json
from pathlib import Path
from datetime import datetime

# Base SAJHA MCP Server structure (from GitHub repo)
BASE_REPO_FILES = {
    # Core files
    "run_server.py",
    "requirements.txt",
    "verify_installation.py",
    "README.md",
    
    # Core module
    "core/__init__.py",
    "core/properties_configurator.py",
    "core/auth_manager.py",
    "core/mcp_handler.py",
    
    # Tools module (base)
    "tools/__init__.py",
    "tools/base_mcp_tool.py",
    "tools/tools_registry.py",
    "tools/impl/__init__.py",
    "tools/impl/wikipedia_tool.py",
    "tools/impl/yahoo_finance_tool.py",
    "tools/impl/google_search_tool.py",
    "tools/impl/fed_reserve_tool.py",
    
    # Web module (base)
    "web/__init__.py",
    "web/app.py",
    "web/templates/base.html",
    "web/templates/login.html",
    "web/templates/dashboard.html",
    "web/templates/error.html",
    "web/templates/tools_list.html",
    "web/templates/tool_execute.html",
    "web/templates/tool_schema.html",
    "web/static/css/style.css",
    "web/static/js/main.js",
    
    # Config (base)
    "config/server.properties",
    "config/application.properties",
    "config/users.json",
    "config/tools/wikipedia.json",
    "config/tools/yahoo_finance.json",
    "config/tools/google_search.json",
    "config/tools/fed_reserve.json",
    
    # Other base
    "logs/.gitkeep",
    "test/__init__.py",
    "docs/README.md",
    "sajha/__init__.py",
}

# Files/patterns to KEEP in place (not move to external)
KEEP_IN_PLACE = {
    "tools/impl/tavily_tool_refactored.py",  # Per user request
    "venv/",
    ".git/",
    "__pycache__/",
    "*.pyc",
    "server.log",
    ".env",
}

# Already in external/ (no action needed)
ALREADY_EXTERNAL = {
    "external/",
}

def scan_project(root_path: str) -> dict:
    """Scan project and categorize all files."""
    results = {
        "base_files": [],      # Files from base repo (stay in place)
        "custom_files": [],    # Files to move to external/
        "already_external": [],# Already in external/
        "keep_in_place": [],   # Custom but should stay (e.g., tavily)
        "ignore": [],          # Temp files, cache, etc.
    }
    
    root = Path(root_path)
    
    for path in root.rglob("*"):
        if path.is_dir():
            continue
            
        rel_path = str(path.relative_to(root))
        
        # Skip ignored patterns
        if any(pattern in rel_path for pattern in ["__pycache__", ".pyc", "venv/", ".git/", ".env", "server.log"]):
            results["ignore"].append(rel_path)
            continue
        
        # Already in external
        if rel_path.startswith("external/"):
            results["already_external"].append(rel_path)
            continue
        
        # Check if base file
        if rel_path in BASE_REPO_FILES:
            results["base_files"].append(rel_path)
            continue
        
        # Check if should keep in place
        if any(keep in rel_path for keep in KEEP_IN_PLACE):
            results["keep_in_place"].append(rel_path)
            continue
        
        # Everything else is custom - should move to external
        results["custom_files"].append(rel_path)
    
    return results

def generate_migration_manifest(results: dict) -> dict:
    """Generate migration manifest with specific actions."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "base_files": len(results["base_files"]),
            "custom_files_to_move": len(results["custom_files"]),
            "already_in_external": len(results["already_external"]),
            "keep_in_place": len(results["keep_in_place"]),
        },
        "actions": []
    }
    
    # Group custom files by directory
    by_dir = {}
    for f in results["custom_files"]:
        dir_name = f.split("/")[0] if "/" in f else "root"
        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(f)
    
    for dir_name, files in sorted(by_dir.items()):
        manifest["actions"].append({
            "source_dir": dir_name,
            "target_dir": f"external/{dir_name}" if dir_name != "root" else "external/",
            "files": files,
            "file_count": len(files),
        })
    
    return manifest

def main():
    print("=" * 60)
    print("MIGRATION AUDIT - SAJHA MCP Server")
    print("=" * 60)
    
    results = scan_project(".")
    manifest = generate_migration_manifest(results)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Base repo files (stay):     {manifest['summary']['base_files']}")
    print(f"   Custom files (to move):     {manifest['summary']['custom_files_to_move']}")
    print(f"   Already in external/:       {manifest['summary']['already_in_external']}")
    print(f"   Keep in place (exceptions): {manifest['summary']['keep_in_place']}")
    
    print(f"\n📁 FILES TO MOVE BY DIRECTORY:")
    for action in manifest["actions"]:
        print(f"\n   {action['source_dir']}/ → {action['target_dir']}/")
        print(f"   Files: {action['file_count']}")
        for f in action['files'][:5]:  # Show first 5
            print(f"      - {f}")
        if len(action['files']) > 5:
            print(f"      ... and {len(action['files']) - 5} more")
    
    # Save full manifest
    with open("migration_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Full manifest saved to: migration_manifest.json")
    
    # Save detailed lists
    with open("migration_details.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MIGRATION DETAILS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BASE FILES (no action):\n")
        for item in sorted(results["base_files"]):
            f.write(f"  ✓ {item}\n")
        
        f.write("\nCUSTOM FILES (to move to external/):\n")
        for item in sorted(results["custom_files"]):
            f.write(f"  → {item}\n")
        
        f.write("\nALREADY IN EXTERNAL (no action):\n")
        for item in sorted(results["already_external"])[:20]:
            f.write(f"  ✓ {item}\n")
        if len(results["already_external"]) > 20:
            f.write(f"  ... and {len(results['already_external']) - 20} more\n")
        
        f.write("\nKEEP IN PLACE (exceptions):\n")
        for item in sorted(results["keep_in_place"]):
            f.write(f"  ⚡ {item}\n")
    
    print(f"✅ Detailed list saved to: migration_details.txt")
    
    return manifest

if __name__ == "__main__":
    main()


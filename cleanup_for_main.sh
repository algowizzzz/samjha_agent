#!/bin/bash
# Cleanup script to sync with main branch
# Removes test files, development docs, and unnecessary files

echo "Starting cleanup process..."

# ============================================================================
# PHASE 1: Delete Test Files
# ============================================================================
echo "Phase 1: Removing test files..."

# Delete test files in test/ directory
rm -f test/test_agent_locking.py
rm -f test/test_bank_of_canada_mcp_tool.py
rm -f test/test_change_selection_plan.py
rm -f test/test_deep_research_agent.py
rm -f test/test_deep_research_tools.py
rm -f test/test_doc_processing_tools.py
rm -f test/test_doc_review_agent.py
rm -f test/test_doc_review_routes.py
rm -f test/test_doc_review_vfs.py
rm -f test/test_duckdb_mcp_tools.py
rm -f test/test_european_central_bank_tool.py
rm -f test/test_fbi_mcp_tool.py
rm -f test/test_federal_reserve_mcp_tool.py
rm -f test/test_google_search_mcp_tool.py
rm -f test/test_imf_mcp_tool.py
rm -f test/test_investor_relations_mcp_tool.py
rm -f test/test_model_doc_api_bypass_auth.py
rm -f test/test_model_doc_api_endpoints.py
rm -f test/test_model_doc_api_endpoints_simple.py
rm -f test/test_model_doc_api_no_auth.py
rm -f test/test_model_doc_routes.py
rm -f test/test_msdoc_relations_mcp_tool.py
rm -f test/test_parquet_agent.py
rm -f test/test_sec_edgar_relations_mcp_tool.py
rm -f test/test_sql_select_relations_mcp_tool.py
rm -f test/test_tavily_mcp_tool.py
rm -f test/test_united_nations_mcp_tool.py
rm -f test/test_webcrawler_mcp_tool.py
rm -f test/test_wikipedia_mcp_tool.py
rm -f test/test_world_bank_mcp_tool.py
rm -f test/test_yahoo_finance_mcp_tool.py

# Delete test documentation files
rm -f test/TEST_RESULTS_DOC_REVIEW.md
rm -f test/TEST_CASES_EXECUTED.md
rm -f test/README_MODEL_DOC_TESTS.md
rm -f test/run_model_doc_tests_with_auth.md
rm -f test/run_model_doc_api_tests.sh
rm -f test/test_doc_review_frontend_websocket.html
rm -f test/test_doc_review_frontend_websocket.js
rm -f test/IDE_E2E_CHECKLIST.txt

# Delete test files in docs/
rm -f docs/test_phase1_tools.py
rm -f docs/test_phase2_tools.py
rm -f docs/test_phase3_tools.py
rm -f docs/test_phase1_expected
rm -f docs/test_phase2_expected
rm -f docs/test_phase2_expected.json
rm -f docs/test_phase3_expected.json
rm -rf docs/test_phase1_results/
rm -rf docs/test_phase2_results/
rm -rf docs/test_phase3_results/

echo "✓ Test files removed"

# ============================================================================
# PHASE 2: Delete Development Documentation
# ============================================================================
echo "Phase 2: Removing development documentation..."

rm -f docs/AGENT_PLANNER_IMPLEMENTATION.md
rm -f docs/agent_state_flow_diagram.md
rm -f docs/CACHE_CLEAR_INSTRUCTIONS.md
rm -f docs/chunking_controls_explanation.md
rm -f docs/DEBUG_PAGE_NOT_LOADING.md
rm -f docs/EXTRACTION_FIX_SUMMARY.md
rm -f docs/FIXES_APPLIED.md
rm -f docs/FULL_WORKFLOW_TEST_RESULTS.md
rm -f docs/IDE_INTEGRATION.md
rm -f docs/PHASE2_VALIDATION_RESULTS.md
rm -f docs/phase3_analysis_workflow.md
rm -f docs/PHASE3_IMPROVEMENTS.md
rm -f docs/ROUTE_FIX.md
rm -f docs/TESTING_NEW_UI.md
rm -f docs/VSCODE_WEB_SETUP.md
rm -f docs/CAR_Chapter_1_Overview.pdf
rm -f docs/examples_query_patterns.md
rm -rf docs/doc-review/

echo "✓ Development docs removed"

# ============================================================================
# PHASE 3: Delete React UI Development Docs
# ============================================================================
echo "Phase 3: Removing React UI development docs..."

rm -f "Doc Review Workspace Wireframe/AUTOSAVE_DISABLED_TRACKING_REMOVED.md"
rm -f "Doc Review Workspace Wireframe/BOLD_AND_FORMATTING_FIX.md"
rm -f "Doc Review Workspace Wireframe/ENHANCED_EDITOR_FEATURES.md"
rm -f "Doc Review Workspace Wireframe/FEATURE_STATUS.md"
rm -f "Doc Review Workspace Wireframe/FEATURES_ADDED.md"
rm -f "Doc Review Workspace Wireframe/THREE_BUTTONS_CONSOLIDATED.md"
rm -f "Doc Review Workspace Wireframe/THREE_CRITICAL_FIXES.md"

echo "✓ React UI dev docs removed"

# ============================================================================
# PHASE 4: Delete IR Documentation (Keep Essential)
# ============================================================================
echo "Phase 4: Cleaning IR documentation..."

rm -f ir/FINAL_SUMMARY.md
rm -f ir/INTEGRATION_GUIDE.md
rm -f ir/MIGRATION.md
rm -f "ir/Investor Relations Tool - Architecture Diagram.md"
rm -f "ir/Investor Relations Tool - Complete Package Index.md"
rm -f "ir/Investor Relations Tool - Executive Summary.md"
rm -f "ir/Investor Relations Tool - Testing Guide.md"

echo "✓ IR docs cleaned (kept README and QUICK_START)"

# ============================================================================
# PHASE 5: Delete Development Scripts
# ============================================================================
echo "Phase 5: Removing development scripts..."

rm -f show_full_structure.py
rm -f show_prompt_monitor.py
rm -f verify_installation.py
rm -f setup_llm.sh
rm -f run_test.sh
rm -f restart_server.sh

echo "✓ Development scripts removed"

# ============================================================================
# PHASE 6: Clean Test Data
# ============================================================================
echo "Phase 6: Cleaning test data..."

rm -rf data/model_doc/test_sample_codebase/
rm -f external/data/doc_review/markdown/test_semantic_blocks_md_*.md

echo "✓ Test data cleaned"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Cleanup Complete!"
echo "============================================"
echo ""
echo "Files removed:"
echo "  - ~34 test files"
echo "  - ~20 development documentation files"
echo "  - ~5 development scripts"
echo "  - Test data directories"
echo ""
echo "Preserved:"
echo "  ✓ All external/ folder (agents, routes, tools)"
echo "  ✓ React UI (Doc Review Workspace Wireframe/)"
echo "  ✓ User guides (docs/userguides/)"
echo "  ✓ Core functionality"
echo "  ✓ Production configuration"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git status"
echo "  2. Commit: git add . && git commit -m 'Cleanup for main sync'"
echo "  3. Push: git push origin all_agents"
echo ""


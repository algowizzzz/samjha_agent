# Model Documentation API Endpoint Tests

This directory contains test scripts for the Model Documentation Agent API endpoints.

## Prerequisites

1. **Server must be running**: The Flask server needs to be running on `http://localhost:5000`
2. **Authentication**: Tests attempt to authenticate using default credentials (admin/admin). Adjust in the script if needed.
3. **Dependencies**: The script requires the `requests` library.

## Quick Start

### Option 1: Using the shell script

```bash
./test/run_model_doc_api_tests.sh
```

### Option 2: Direct Python execution

```bash
python3 test/test_model_doc_api_endpoints.py
```

## What the Tests Do

The test script:

1. **Checks server connectivity** - Verifies the server is running
2. **Authenticates** - Attempts to log in and get an auth token
3. **Creates a sample codebase** - Generates a test Python codebase in `data/model_doc/test_sample_codebase/`
4. **Tests all API endpoints**:
   - `GET /api/model_doc/templates` - List available templates
   - `POST /api/model_doc/documents` - Register a new codebase
   - `GET /api/model_doc/documents` - List all codebases
   - `GET /api/model_doc/documents/<codebase_id>` - Get codebase details
   - `GET /api/model_doc/documents/<codebase_id>/status` - Get workflow status
   - `PATCH /api/model_doc/documents/<codebase_id>/config` - Update configuration
   - `POST /api/model_doc/documents/<codebase_id>/run_phase1` - Run Phase 1 (Codebase Discovery)
   - `POST /api/model_doc/chat/<codebase_id>` - Chat with codebase (requires LLM)

## Sample Codebase

The test creates a sample codebase with:

- `calculator.py` - A Calculator class with basic arithmetic operations
- `utils/helpers.py` - Utility helper functions
- `README.md` - Sample documentation

## Running Specific Tests

You can modify the `main()` function in `test_model_doc_api_endpoints.py` to comment out specific tests or add new ones.

## Expected Output

When tests run successfully, you should see:

```
======================================================================
Model Documentation Agent - API Endpoint Tests
======================================================================

🔍 Checking if server is running at http://localhost:5000...
   ✅ Server is running

🔐 Authenticating...
   ✅ Authentication successful

📦 Creating sample codebase...
   ✅ Created sample codebase at: /path/to/data/model_doc/test_sample_codebase

🔍 Testing: GET /api/model_doc/templates
   Status: 200
   ✅ Found 3 templates

🔍 Testing: POST /api/model_doc/documents
   Status: 201
   ✅ Registered codebase: test_calculator_codebase

... (more tests)

✅ All endpoint tests completed!
```

## Troubleshooting

### Server not running
- Make sure the Flask server is started: `python3 run_server.py` or `python3 -m flask run`

### Authentication fails
- Check your server's authentication setup
- Modify `TEST_USERNAME` and `TEST_PASSWORD` in the script if needed
- Some endpoints may work without authentication depending on your setup

### Import errors
- Make sure you're running from the project root directory
- Install required dependencies: `pip install requests`

### LLM endpoints fail
- The chat endpoint requires LLM to be configured
- Phase 2+ endpoints require LLM for summarization
- These tests will show warnings but won't fail the test suite

## Full Workflow Test

By default, the full workflow test is commented out because it:
- Takes several minutes to complete
- Requires LLM to be configured
- Generates complete documentation

To enable it, uncomment this line in `main()`:
```python
test_run_full_workflow(token, codebase_id)
```

## Test Data Location

Sample codebase is created at:
```
data/model_doc/test_sample_codebase/
```

You can manually inspect or modify this codebase for testing purposes.


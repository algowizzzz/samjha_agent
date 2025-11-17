# Running Model Documentation API Tests

The API endpoints require authentication via Flask sessions. Here's how to test them:

## Quick Start (Manual Authentication)

### Step 1: Start the Server
```bash
python3 run_server.py
```

### Step 2: Login via Browser
1. Open browser and go to: http://localhost:5000/login
2. Login with:
   - User ID: `admin`
   - Password: `admin123`
3. After successful login, open browser DevTools (F12)
4. Go to Application/Storage → Cookies → http://localhost:5000
5. Copy the `session` cookie value

### Step 3: Run Tests with Cookie

```bash
# Set the session cookie
export FLASK_SESSION_COOKIE="<paste_cookie_value_here>"

# Run the test script
python3 test/test_model_doc_api_endpoints.py
```

Or modify the test script to include the cookie directly.

## Alternative: Use Browser Extension

Use a browser extension like "Cookie Editor" to export cookies as JSON, then load them in the test script.

## Expected Test Results

When properly authenticated, you should see:

1. ✅ List templates
2. ✅ Register codebase
3. ✅ List codebases  
4. ✅ Get codebase details
5. ✅ Get status
6. ✅ Update config
7. ✅ Run Phase 1 (discovers files, parses AST, builds hierarchy)
8. ⚠️  Chat endpoint (may fail if LLM not configured)

## Sample Codebase

The test creates a sample codebase at:
```
data/model_doc/test_sample_codebase/
```

Contains:
- `calculator.py` - Calculator class
- `utils/helpers.py` - Helper functions


# Route Registration Fix

## Problem
The document review routes were returning 404 errors because route registration was failing due to a function name conflict.

## Error
```
ERROR:root:Failed to register doc review routes: View function mapping is overwriting an existing endpoint function: get_welcome_message
```

## Solution
Renamed the conflicting function from `get_welcome_message()` to `doc_review_get_welcome_message()` in `external/routes/doc_review_routes.py`.

## Verification
After the fix, all routes are now registered:
- ✓ `/api/doc_review/documents` - Found
- ✓ `/api/doc_review/upload` - Found
- ✓ `/api/doc_review/token` - Found
- ✓ `/api/doc_review/welcome` - Found
- ✓ All other doc_review routes - Found

## Next Steps
**RESTART THE SERVER** to apply the fix:

```bash
# Kill existing server
pkill -f "python.*run_server"

# Start server with venv
cd /Users/saadahmed/samjha_agent/samjha_agent
source venv/bin/activate
python3 run_server.py
```

After restarting, the 404 errors should be resolved and the upload/API calls should work.


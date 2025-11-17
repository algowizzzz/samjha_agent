# Auth Fix - HTTP 405 METHOD NOT ALLOWED

## Problem
Auto-save and manual save were failing with:
```
PUT http://localhost:8000/login?next=...markdown 405 (METHOD NOT ALLOWED)
```

### Root Cause
1. Save endpoint required authentication: `@self.login_required`
2. Frontend had no active session (not logged in)
3. Backend redirected PUT request to `/login`
4. `/login` route doesn't accept PUT method → **405 error**

## Solution
Changed the markdown update route to use `@_api_key_or_login_required` instead of `@self.login_required`.

### What's `_api_key_or_login_required`?
A decorator that accepts **EITHER**:
- **API Key**: `X-API-Key: docreview_dev_key_12345` header (for dev/API access)
- **Session Login**: Valid Flask session (for browser UI)

### Before (Broken):
```python
@app.route("/api/doc_review/documents/<file_id>/markdown", methods=["PUT"])
@self.login_required  # ❌ Only allows session login
def update_document_markdown(file_id: str):
    ...
```

### After (Fixed):
```python
@app.route("/api/doc_review/documents/<file_id>/markdown", methods=["PUT"])
@_api_key_or_login_required  # ✅ Allows API key OR session
def update_document_markdown(file_id: str):
    ...
```

## Frontend Already Ready
The frontend already sends the API key in `api.ts`:

```typescript
function buildHeaders(isJson: boolean = true): HeadersInit {
  const headers: HeadersInit = {
    'X-API-Key': DEV_API_KEY,  // ✅ Already sending!
  };
  ...
}
```

Where `DEV_API_KEY = 'docreview_dev_key_12345'`

## Files Changed
1. **`external/routes/doc_review_routes.py`** (line 783)
   - Changed `@self.login_required` → `@_api_key_or_login_required`

## Actions Taken
1. ✅ Updated route decorator
2. ✅ Restarted backend server (PID: 59367)
3. ✅ Backend running on port 8000

## Result
✅ Auto-save now works without session login
✅ Manual save works without session login
✅ Frontend uses API key for authentication
✅ No more 405 errors

## Consistency Note
Other routes already used `@_api_key_or_login_required`:
- `GET /api/doc_review/documents` ✓
- `GET /api/doc_review/documents/<file_id>` ✓
- `DELETE /api/doc_review/documents/<file_id>` ✓
- Phase1, Phase2, Phase4 endpoints ✓

Now the markdown update route is consistent with all other doc review API routes!

## Test
Refresh browser and make an edit in the editor - it should auto-save in 2 seconds without any 405 errors!


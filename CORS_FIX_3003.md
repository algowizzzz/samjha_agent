# CORS Fix for Port 3003

## Problem
Frontend moved from port `3002` to port `3003`, but backend CORS policy only allowed `3002`.

Error message:
```
Access to fetch at 'http://localhost:8000/api/doc_review/documents' from origin 'http://localhost:3003' 
has been blocked by CORS policy: Response to preflight request doesn't pass access control check: 
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## Solution
Added port `3003` to CORS allowed origins in `web/app.py`.

### Before:
```python
origins=["http://localhost:3002", "http://localhost:3001", "http://localhost:3000", ...]
```

### After:
```python
origins=["http://localhost:3003", "http://localhost:3002", "http://localhost:3001", "http://localhost:3000", ...]
```

## File Changed
- `web/app.py` (line 77)

## Actions Taken
1. ✅ Added `http://localhost:3003` and `http://127.0.0.1:3003` to CORS origins
2. ✅ Restarted backend server (PID: 55406)
3. ✅ Backend running on port 8000

## Status
✅ CORS error resolved
✅ Frontend (port 3003) can now communicate with backend (port 8000)

## Ports Now Supported
- `http://localhost:3003` ← **New**
- `http://localhost:3002`
- `http://localhost:3001`
- `http://localhost:3000`
- `http://127.0.0.1:3003` ← **New**
- `http://127.0.0.1:3002`
- `http://127.0.0.1:3001`
- `http://127.0.0.1:3000`

## Test
Refresh your browser at `http://localhost:3003` - documents list should load without CORS errors!


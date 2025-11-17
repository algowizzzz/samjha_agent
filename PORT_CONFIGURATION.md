# Port Configuration Fixed 🔧

## Issue
macOS ControlCenter was using port 5000, preventing Flask from binding to it.

## Solution
Flask server now runs on **port 8000** instead of port 5000.

## Updated URLs

### Backend API
```
http://localhost:8000
```

### Frontend Dev Server
```
http://localhost:3001
```
(Note: Port 3000 was in use, so Vite used 3001)

## New API Endpoints Working ✅

### Prompts API
- **List all prompts**: `GET http://localhost:8000/api/doc_review/prompts`
- **Get prompt**: `GET http://localhost:8000/api/doc_review/prompts/<name>`
- **Update prompt**: `PUT http://localhost:8000/api/doc_review/prompts/<name>`

### Templates API
- **List templates**: `GET http://localhost:8000/api/doc_review/templates`
- **Upload template**: `POST http://localhost:8000/api/doc_review/templates/upload`

## Test Results

```bash
$ curl http://localhost:8000/api/doc_review/prompts
{
  "prompts": [
    {"name": "content_improvement", "filename": "content_improvement.txt", "size": 2215},
    {"name": "gap_analysis", "filename": "gap_analysis.txt", "size": 1725}
  ]
}
```

```bash
$ curl http://localhost:8000/api/doc_review/templates  
{
  "templates": ["policy_template"]
}
```

## Frontend Configuration Updated

File: `Doc Review Workspace Wireframe/src/lib/api.ts`

The frontend now automatically uses:
- **Development**: `http://localhost:8000/api`
- **Production**: `/api` (relative path)

## Access Application

Open your browser to:
```
http://localhost:3001
```

The Prompts and Templates pages should now work correctly! 🎉


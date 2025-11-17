#!/usr/bin/env python3
"""
Simple IDE/VFS smoke: lists documents, picks one, hits VFS endpoints.
Run with project venv:
  ./venv/bin/python scripts/ide_smoke.py
"""
import os
import sys
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

BASE = os.environ.get("DOCREVIEW_BASE_URL", "http://localhost:8000")


def _get(path, params=None):
    qs = ""
    if params:
        qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}{path}{qs}"
    req = Request(url, method="GET")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _patch(path, data):
    url = f"{BASE}{path}"
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, method="PATCH")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    try:
        docs = _get("/api/doc_review/documents")
        records = docs.get("documents", [])
        if not records:
            print("NO_DOCUMENTS")
            return 0
        first = records[0]
        file_id = first.get("file_id")
        if not file_id:
            print("INVALID_DOC_LIST")
            return 1
        # tree stat file
        tree = _get("/api/doc_review/vfs/tree", {"file_id": file_id, "path": "/"})
        stat = _get("/api/doc_review/vfs/stat", {"file_id": file_id, "path": "/phase1"})
        # try read any known file if present
        content_ok = True
        try:
            f = _get("/api/doc_review/vfs/file", {"file_id": file_id, "path": "/phase1/doc_summary.json"})
            _ = f.get("content", "")
        except Exception:
            content_ok = False
        print("OK", file_id, len(tree.get("entries", [])), stat.get("stat", {}).get("type"), "CONTENT" if content_ok else "NO_CONTENT")
        return 0
    except HTTPError as e:
        print("HTTP_ERROR", e.code, e.reason)
        return 2
    except URLError as e:
        print("URL_ERROR", getattr(e, "reason", "")) 
        return 3
    except Exception as e:
        print("ERROR", str(e))
        return 4


if __name__ == "__main__":
    sys.exit(main())



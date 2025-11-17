# Debug: Page Not Loading

If the page doesn't load, follow these steps to identify the issue:

## Step 1: Check Server Logs

When you try to access `/doc-review`, check the server terminal output for errors:

```bash
# Look for errors like:
# - TemplateNotFound
# - SyntaxError
# - ImportError
# - 500 Internal Server Error
```

## Step 2: Check Browser Console

1. Open browser DevTools (F12)
2. Go to **Console** tab
3. Look for JavaScript errors (red text)
4. Common errors:
   - `Uncaught ReferenceError: X is not defined`
   - `Uncaught TypeError: Cannot read property 'X' of null`
   - `Failed to load resource: /static/css/doc_review_cockpit.css`

## Step 3: Check Network Tab

1. Open browser DevTools (F12)
2. Go to **Network** tab
3. Refresh the page
4. Look for:
   - Failed requests (red)
   - 404 errors (CSS/JS files not found)
   - 500 errors (server errors)

## Step 4: Verify Files Exist

Run these commands:

```bash
# Check template exists
ls -la web/templates/doc_review_cockpit_new.html

# Check CSS exists
ls -la web/static/css/doc_review_cockpit.css

# Check welcome message exists
ls -la config/agent_welcome.md
```

## Step 5: Test Template Rendering

If you see a blank page, the template might be rendering but JavaScript is failing.

**Quick Test:**
1. Right-click on page → "View Page Source"
2. If you see HTML content, the template is rendering
3. If you see nothing or an error message, it's a server-side issue

## Step 6: Common Issues & Fixes

### Issue: Blank White Page
**Possible Causes:**
- JavaScript error preventing render
- CSS not loading (page renders but invisible)
- Template syntax error

**Fix:**
- Check browser console for errors
- Verify CSS file loads: `http://localhost:5000/static/css/doc_review_cockpit.css`
- Check server logs for template errors

### Issue: 500 Internal Server Error
**Possible Causes:**
- Template syntax error
- Missing import
- Route handler error

**Fix:**
- Check server terminal for full error traceback
- Verify template extends base.html correctly
- Check that all required routes are registered

### Issue: 404 Not Found
**Possible Causes:**
- Route not registered
- Template file not found
- Static file not found

**Fix:**
- Verify route exists: `@app.route("/doc-review")`
- Check template file path is correct
- Verify static files are in correct location

### Issue: JavaScript Errors in Console
**Common Errors:**

1. **`Cannot read property 'X' of null`**
   - Element not found in DOM
   - **Fix:** Already fixed - elements now initialize after DOM loads

2. **`marked is not defined`**
   - Marked.js library not loaded
   - **Fix:** Check CDN link in template

3. **`fetch failed`**
   - API endpoint not available
   - **Fix:** Check server is running and routes are registered

## Step 7: Quick Diagnostic Commands

```bash
# Test if server is running
curl http://localhost:5000/doc-review

# Check if template file is readable
cat web/templates/doc_review_cockpit_new.html | head -20

# Check if CSS file exists
ls -la web/static/css/doc_review_cockpit.css

# Test Python syntax
python3 -m py_compile web/templates/doc_review_cockpit_new.html 2>&1 || echo "Not a Python file (expected)"
```

## Step 8: Fallback to Old Template

If the new template has issues, you can temporarily use the old one:

1. Edit `external/routes/doc_review_routes.py`
2. Change line 68 from:
   ```python
   return render_template("doc_review_cockpit_new.html")
   ```
   To:
   ```python
   return render_template("doc_review_cockpit.html")
   ```

## Step 9: Enable Debug Mode

Add this to see more detailed errors:

```python
# In run_server.py or app.py
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
```

## What to Report

If the page still doesn't load, please provide:

1. **Browser Console Errors** (screenshot or copy-paste)
2. **Server Terminal Output** (error messages)
3. **Network Tab** (failed requests)
4. **Page Source** (right-click → View Source → first 50 lines)

This will help identify the exact issue.


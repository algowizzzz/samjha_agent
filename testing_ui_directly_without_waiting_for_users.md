# Testing UI Directly Without Waiting for Users

A guide for AI coding agents in Cursor to test web UIs programmatically using MCP Browser Tools.

## Available Browser Tools

| Tool | Purpose |
|------|---------|
| `mcp_cursor-ide-browser_browser_navigate` | Navigate to a URL |
| `mcp_cursor-ide-browser_browser_snapshot` | Get page accessibility tree (elements + refs) |
| `mcp_cursor-ide-browser_browser_click` | Click on an element |
| `mcp_cursor-ide-browser_browser_type` | Type text into inputs |
| `mcp_cursor-ide-browser_browser_wait_for` | Wait for time or text to appear/disappear |
| `mcp_cursor-ide-browser_browser_console_messages` | Get JS console logs/errors |
| `mcp_cursor-ide-browser_browser_press_key` | Press keyboard keys |
| `mcp_cursor-ide-browser_browser_take_screenshot` | Capture visual screenshot |
| `mcp_cursor-ide-browser_browser_hover` | Hover over an element |
| `mcp_cursor-ide-browser_browser_select_option` | Select dropdown option |
| `mcp_cursor-ide-browser_browser_navigate_back` | Go back to previous page |
| `mcp_cursor-ide-browser_browser_network_requests` | View network requests |

## Basic Workflow

### Step 1: Navigate to the Page

```
browser_navigate → url: "http://localhost:8000/login"
```

This opens the browser and navigates to the specified URL. The response includes a page snapshot.

### Step 2: Get Page Snapshot

```
browser_snapshot
```

Returns a YAML accessibility tree with:
- Element roles (button, textbox, link, etc.)
- Element names/labels
- **`ref` IDs** - unique identifiers needed for interactions

Example output:
```yaml
- role: textbox
  name: User ID
  ref: ref-abc123
- role: button
  name: Sign In
  ref: ref-xyz789
```

### Step 3: Interact with Elements

**Type into input fields:**
```
browser_type → element: "User ID input", ref: "ref-abc123", text: "admin"
```

**Click buttons:**
```
browser_click → element: "Sign In button", ref: "ref-xyz789"
```

**Submit forms (press Enter after typing):**
```
browser_type → element: "Password", ref: "ref-pwd456", text: "admin123", submit: true
```

### Step 4: Wait for Page Changes

```
browser_wait_for → time: 2
```

Wait 2 seconds for page to load/update. Always wait after clicks or form submissions.

### Step 5: Check for JavaScript Errors

```
browser_console_messages
```

**This is critical for debugging!** Returns all console logs, warnings, and errors:
```json
{
  "type": "debug",
  "message": "Uncaught SyntaxError: Unexpected token ')' (line 1706)"
}
```

## Complete Login Flow Example

```
1. browser_navigate 
   → url: "http://localhost:8000/login"

2. browser_type 
   → element: "User ID input"
   → ref: "<from-snapshot>"
   → text: "admin"

3. browser_type 
   → element: "Password input"
   → ref: "<from-snapshot>"
   → text: "admin123"
   → submit: true

4. browser_wait_for 
   → time: 2

5. browser_snapshot
   (verify you're on the expected page)

6. browser_console_messages
   (check for any JS errors)
```

## Debugging Tips

### 1. Always Check Console After Page Load
JavaScript syntax errors will appear in `browser_console_messages`. This is often the fastest way to find why buttons don't work.

### 2. Refs Are Ephemeral
Element refs change after each page navigation or significant DOM update. Always get fresh refs with `browser_snapshot` after:
- Page navigation
- Form submissions
- AJAX updates
- Modal opens/closes

### 3. Re-snapshot After Interactions
The page state changes after clicks and form submissions. Call `browser_snapshot` again to see the new state.

### 4. Use Descriptive Element Names
When calling `browser_click` or `browser_type`, the `element` parameter is a human-readable description for logging purposes. The `ref` is what actually identifies the element.

### 5. Handle Authentication
Most apps require login. Test the login flow first, then navigate to protected pages while the session is active.

## Real-World Debugging Example

### Problem
User reported: "Send button doesn't work"

### Solution Process

1. **Navigate to the page:**
   ```
   browser_navigate → url: "http://localhost:8000/agent/chat/ecommerce_advanced"
   ```

2. **Check console immediately:**
   ```
   browser_console_messages
   ```

3. **Found the error:**
   ```
   "Uncaught SyntaxError: Unexpected token ')' (line 1708)"
   ```

4. **Root cause identified:**
   A missing closing brace `}` in the JavaScript caused the entire script to fail, which meant no event listeners were attached to buttons.

5. **Fix applied:**
   Added the missing `}` to close the `setupEventListeners` function.

6. **Verified fix:**
   - Restarted server
   - Re-navigated to page
   - Checked console (no errors)
   - Tested button clicks (working)

## When to Use These Tools

✅ **Use browser tools when:**
- User reports UI bugs (buttons not working, pages not loading)
- Need to verify frontend changes without user involvement
- Debugging JavaScript errors
- Testing authentication flows
- Verifying form submissions

❌ **Don't use browser tools when:**
- Backend-only changes (use terminal/API tests instead)
- Simple file edits that don't affect UI
- User hasn't provided the URL/port

## Key Insight

The `browser_console_messages` tool is the most valuable for debugging. It reveals JavaScript errors that would otherwise require the user to open browser dev tools and report back - saving significant back-and-forth time.


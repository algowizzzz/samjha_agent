# Frontend Debugging Guide

## Common Issues & Solutions

### 1. **Button/Element Not Showing Up**

**Symptoms:**
- Added a button in HTML but it doesn't appear in browser
- Changed CSS but changes don't show

**Debugging Steps:**

1. **Check Browser Cache:**
   ```javascript
   // In browser console, check version:
   console.log('Version:', window.AGENT_CHAT_VERSION);
   console.log('Build:', window.AGENT_CHAT_BUILD);
   ```
   - If version is wrong, do **hard refresh**: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
   - Or clear cache: DevTools → Application → Clear Storage

2. **Check HTML is Actually Loaded:**
   ```javascript
   // In browser console:
   document.getElementById('showThinkingToggle')  // Should return element, not null
   ```
   - If `null`, HTML wasn't loaded or ID is wrong

3. **Check CSS is Applied:**
   ```javascript
   // In browser console:
   const el = document.getElementById('showThinkingToggle');
   window.getComputedStyle(el).display;  // Should not be 'none'
   ```

4. **Check JavaScript Errors:**
   - Open DevTools Console (F12)
   - Look for red errors
   - Check if your code is even running

5. **Check Element is in DOM:**
   ```javascript
   // In browser console:
   document.querySelector('#showThinkingToggle')  // Should find it
   ```

### 2. **JavaScript Not Running**

**Symptoms:**
- Event handlers don't fire
- Variables are undefined
- Functions don't execute

**Debugging Steps:**

1. **Check Script is Loaded:**
   ```javascript
   // In browser console:
   typeof API  // Should be 'object', not 'undefined'
   typeof $     // Should be 'function' (jQuery)
   ```

2. **Check for Syntax Errors:**
   - Open DevTools Console
   - Look for syntax errors (red text)
   - Check line numbers

3. **Check Execution Order:**
   ```javascript
   // Add console.log at start of function:
   function sendQuery() {
       console.log('[DEBUG] sendQuery called');  // Should see this in console
       // ... rest of code
   }
   ```

4. **Check Event Listeners:**
   ```javascript
   // In browser console:
   const btn = document.getElementById('sendBtn');
   btn.addEventListener('click', () => console.log('Clicked!'));  // Test if events work
   ```

### 3. **API Calls Not Working**

**Symptoms:**
- No response from backend
- Errors in network tab
- Data not showing

**Debugging Steps:**

1. **Check Network Tab:**
   - Open DevTools → Network tab
   - Send a query
   - Look for `/api/tools/execute` request
   - Check:
     - Status code (should be 200)
     - Request payload (check `show_thinking` is included)
     - Response body (check `final_output.thinking` exists)

2. **Check Request Payload:**
   ```javascript
   // Add before API.post:
   console.log('[DEBUG] Request args:', requestArgs);
   ```

3. **Check Response:**
   ```javascript
   // In .done() callback:
   console.log('[DEBUG] Full response:', resp);
   console.log('[DEBUG] Thinking:', resp.result?.final_output?.thinking);
   ```

4. **Check Authentication:**
   ```javascript
   // In browser console:
   getSessionToken()  // Should return a token string
   ```

### 4. **Thinking Panel Not Showing**

**Symptoms:**
- Toggle is checked
- Backend returns thinking
- But panel doesn't appear

**Debugging Steps:**

1. **Check Toggle State:**
   ```javascript
   // In browser console:
   document.getElementById('showThinkingToggle').checked  // Should be true
   ```

2. **Check Response Has Thinking:**
   ```javascript
   // In .done() callback, add:
   console.log('[DEBUG] Thinking in response:', result.final_output?.thinking);
   console.log('[DEBUG] Thinking length:', result.final_output?.thinking?.length);
   ```

3. **Check Panel Creation:**
   ```javascript
   // In finalizeStreamingMessage, add:
   console.log('[DEBUG] messageEl:', messageEl);
   console.log('[DEBUG] Panel created:', panel);
   console.log('[DEBUG] Panel appended:', messageEl.contains(panel));
   ```

4. **Check CSS:**
   ```javascript
   // In browser console:
   const panel = document.querySelector('.thinking-panel');
   if (panel) {
       console.log('Panel found:', panel);
       console.log('Display:', window.getComputedStyle(panel).display);
       console.log('Visibility:', window.getComputedStyle(panel).visibility);
   }
   ```

### 5. **Systematic Debugging Workflow**

**Step-by-Step Process:**

1. **Open DevTools** (F12)
   - Console tab: Check for errors
   - Network tab: Check API calls
   - Elements tab: Inspect DOM

2. **Add Debug Logs:**
   ```javascript
   // At key points:
   console.log('[DEBUG] Point 1:', variable);
   console.log('[DEBUG] Point 2:', anotherVariable);
   ```

3. **Test Incrementally:**
   - Test toggle checkbox first
   - Test API call second
   - Test rendering third

4. **Verify Data Flow:**
   ```
   User clicks toggle → 
   sendQuery() reads toggle → 
   API.post() sends show_thinking → 
   Backend returns thinking → 
   finalizeStreamingMessage() renders panel
   ```

5. **Check Each Step:**
   - Step 1: `console.log('Toggle:', showThinking)`
   - Step 2: `console.log('Request:', requestArgs)`
   - Step 3: `console.log('Response:', resp)`
   - Step 4: `console.log('Rendering panel')`

### 6. **Quick Debugging Commands**

**Paste in Browser Console:**

```javascript
// Check if toggle exists and is checked
const toggle = document.getElementById('showThinkingToggle');
console.log('Toggle exists:', !!toggle);
console.log('Toggle checked:', toggle?.checked);

// Check if API helper exists
console.log('API exists:', typeof API !== 'undefined');

// Check last response
console.log('Last response:', window.lastResponse);  // Add this to store responses

// Check if thinking panel exists
const panel = document.querySelector('.thinking-panel');
console.log('Thinking panel exists:', !!panel);
if (panel) {
    console.log('Panel visible:', panel.offsetParent !== null);
    console.log('Panel content length:', panel.querySelector('pre')?.textContent?.length);
}

// Force show thinking panel (for testing)
const msg = document.querySelector('.chat-message.agent:last-child');
if (msg) {
    const testPanel = document.createElement('details');
    testPanel.className = 'thinking-panel mt-2';
    testPanel.open = true;
    testPanel.innerHTML = '<summary>🧠 Test Thinking</summary><pre>This is a test</pre>';
    msg.appendChild(testPanel);
    console.log('Test panel added');
}
```

### 7. **Common Fixes**

**Cache Issues:**
- Hard refresh: `Cmd+Shift+R` / `Ctrl+Shift+R`
- Clear cache: DevTools → Application → Clear Storage
- Disable cache: DevTools → Network → "Disable cache" checkbox

**JavaScript Not Running:**
- Check for syntax errors in console
- Check script is loaded (Network tab)
- Check jQuery is loaded: `typeof $`

**Elements Not Showing:**
- Check CSS: `display: none` or `visibility: hidden`
- Check z-index (might be behind other elements)
- Check parent container has height/width

**API Issues:**
- Check authentication token
- Check CORS errors
- Check network tab for failed requests
- Check backend logs

### 8. **Proactive Debugging**

**Add to Your Code:**

```javascript
// At start of sendQuery:
console.log('[DEBUG] ===== SEND QUERY START =====');
console.log('[DEBUG] Query:', q);
console.log('[DEBUG] Show thinking:', showThinking);
console.log('[DEBUG] Toggle element:', document.getElementById('showThinkingToggle'));

// In API response:
console.log('[DEBUG] ===== API RESPONSE =====');
console.log('[DEBUG] Full response:', resp);
console.log('[DEBUG] Result:', result);
console.log('[DEBUG] Final output:', result.final_output);
console.log('[DEBUG] Thinking:', result.final_output?.thinking);

// In finalizeStreamingMessage:
console.log('[DEBUG] ===== FINALIZE MESSAGE =====');
console.log('[DEBUG] Message element:', messageEl);
console.log('[DEBUG] Result:', result);
console.log('[DEBUG] Thinking to render:', thinking);
```

**Then in Browser:**
- Open Console
- Send a query
- Follow the debug logs step-by-step
- Identify where it breaks

### 9. **Testing Checklist**

Before reporting an issue, verify:

- [ ] Hard refreshed browser (`Cmd+Shift+R`)
- [ ] Checked browser console for errors
- [ ] Checked Network tab for API call
- [ ] Verified request includes `show_thinking: true`
- [ ] Verified response includes `final_output.thinking`
- [ ] Verified thinking panel is created in DOM
- [ ] Verified thinking panel is visible (not `display: none`)
- [ ] Tested in incognito/private window (rules out extensions)

### 10. **Getting Help**

When asking for help, provide:

1. **Browser Console Output:**
   - Copy all `[DEBUG]` logs
   - Copy any error messages

2. **Network Tab Screenshot:**
   - Show the `/api/tools/execute` request
   - Show request payload
   - Show response body

3. **DOM Inspection:**
   - Screenshot of Elements tab showing thinking panel
   - Computed styles for the panel

4. **Steps to Reproduce:**
   - What you clicked
   - What you expected
   - What actually happened


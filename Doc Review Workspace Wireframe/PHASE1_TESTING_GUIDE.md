# Phase 1 Testing Guide - Lexical Integration

## 🚀 Quick Start

### Prerequisites
- ✅ Backend running on `http://localhost:8000`
- ✅ Frontend dev server running on `http://localhost:5173`

### Start Frontend (if not running)
```bash
cd "Doc Review Workspace Wireframe"
npm run dev
```

### Access the Application
1. Open browser: `http://localhost:5173`
2. Login if required (via backend at `http://localhost:8000`)

---

## ✅ Test Checklist

### 1. Basic Editor Functionality

#### Test: Text Editing
- [ ] Open a document in the workspace
- [ ] Click on any block to start editing
- [ ] Type text - should work smoothly
- [ ] **Expected**: Text appears as you type, no lag

#### Test: Block Creation
- [ ] Place cursor at end of a block
- [ ] Press `Enter`
- [ ] **Expected**: New paragraph block created below
- [ ] Type in the new block - should work immediately

#### Test: Block Deletion
- [ ] Create a block with some text
- [ ] Delete all text (Backspace)
- [ ] Press `Backspace` again when block is empty
- [ ] **Expected**: Block gets deleted (if not the last block)

---

### 2. Keyboard Shortcuts (Formatting)

#### Test: Bold (Cmd+B / Ctrl+B)
- [ ] Select some text in a block
- [ ] Press `Cmd+B` (Mac) or `Ctrl+B` (Windows/Linux)
- [ ] **Expected**: Text becomes bold
- [ ] Press again - text should unbold

#### Test: Italic (Cmd+I / Ctrl+I)
- [ ] Select text
- [ ] Press `Cmd+I` or `Ctrl+I`
- [ ] **Expected**: Text becomes italic
- [ ] Press again - text should un-italic

#### Test: Underline (Cmd+U / Ctrl+U)
- [ ] Select text
- [ ] Press `Cmd+U` or `Ctrl+U`
- [ ] **Expected**: Text becomes underlined
- [ ] Press again - underline should remove

---

### 3. Block Types

#### Test: Heading 1
- [ ] Type `/` in a paragraph to open slash menu (if implemented)
- [ ] Or manually set block type to heading1
- [ ] **Expected**: Text renders larger, bold
- [ ] Type text - should look like a main heading

#### Test: Heading 2 & 3
- [ ] Switch block to heading2
- [ ] **Expected**: Medium-sized heading
- [ ] Switch to heading3
- [ ] **Expected**: Smaller heading

#### Test: Lists (Bullet & Numbered)
- [ ] Create a bullet list block
- [ ] Type text - **Expected**: Bullet point appears
- [ ] Press `Enter` - **Expected**: New bullet item created
- [ ] Switch to numbered list
- [ ] **Expected**: Numbers instead of bullets

#### Test: Quote
- [ ] Change block to quote type
- [ ] Type text
- [ ] **Expected**: Text appears with left border and italic styling

#### Test: Code
- [ ] Change block to code type
- [ ] Type `const x = 1;`
- [ ] **Expected**: Monospace font, gray background

---

### 4. Editor Behavior

#### Test: Undo/Redo
- [ ] Type some text
- [ ] Press `Cmd+Z` or `Ctrl+Z`
- [ ] **Expected**: Text disappears (undo)
- [ ] Press `Cmd+Shift+Z` or `Ctrl+Shift+Z`
- [ ] **Expected**: Text reappears (redo)

#### Test: Focus Management
- [ ] Click on different blocks
- [ ] **Expected**: Cursor appears in clicked block
- [ ] Tab between blocks (if supported)
- [ ] **Expected**: Focus moves smoothly

#### Test: Drag & Drop (Existing Feature)
- [ ] Hover over a block's drag handle (grip icon)
- [ ] Drag block up/down
- [ ] **Expected**: Block reorders, editor still works after

---

### 5. Integration with Existing Features

#### Test: AI Suggestions
- [ ] Select a block
- [ ] Click "Ask RiskGPT" or use AI suggestion feature
- [ ] **Expected**: Suggestion appears, can accept/reject
- [ ] Editor should still work after accepting suggestion

#### Test: Block Selection
- [ ] Click block's left gutter (checkbox area)
- [ ] **Expected**: Block gets selected (blue highlight)
- [ ] Select multiple blocks (Cmd+Click)
- [ ] **Expected**: Multiple blocks selected
- [ ] Editor still editable in selected blocks

#### Test: Comments
- [ ] Click comment button on a block
- [ ] **Expected**: Comment panel opens
- [ ] Editor should remain functional while comment panel open

---

### 6. Visual Verification

#### Check: No contentEditable Warnings
- [ ] Open browser DevTools (F12)
- [ ] Check Console tab
- [ ] **Expected**: No warnings about `contentEditable` or `execCommand`
- [ ] **Expected**: No errors related to Lexical

#### Check: Styles Applied
- [ ] Format text as bold
- [ ] Inspect element (right-click → Inspect)
- [ ] **Expected**: See CSS class `lexical-bold` applied
- [ ] Check other formats (italic, underline) - should have `lexical-italic`, `lexical-underline`

#### Check: Block Styling
- [ ] Create heading1 block
- [ ] Inspect element
- [ ] **Expected**: See `lexical-h1` class
- [ ] Check list items - should have `lexical-ul` or `lexical-ol`

---

### 7. Performance Test

#### Test: Large Document
- [ ] Create 20+ blocks
- [ ] Type and edit in different blocks
- [ ] **Expected**: No lag, smooth scrolling
- [ ] Drag blocks around
- [ ] **Expected**: No performance issues

#### Test: Rapid Typing
- [ ] Type quickly in a block
- [ ] **Expected**: All characters appear correctly
- [ ] No skipped characters or delays

---

## 🐛 Known Issues / Limitations

### What's NOT Working Yet (Expected):
- ❌ HTML formatting not fully preserved yet (strips HTML on load)
- ❌ Table blocks not supported yet
- ❌ Checkbox blocks not supported yet  
- ❌ Callout blocks not supported yet
- ❌ Full markdown ↔ Lexical conversion (basic only)

### What SHOULD Work:
- ✅ Basic text editing
- ✅ Keyboard shortcuts (Bold, Italic, Underline)
- ✅ Block types (paragraph, headings, lists, quote, code)
- ✅ Enter to create new blocks
- ✅ Backspace to delete empty blocks
- ✅ Undo/Redo
- ✅ Drag & drop reordering
- ✅ Integration with existing features (AI, comments, selection)

---

## 🔍 Debugging

### If Editor Doesn't Work:

1. **Check Console Errors**
   ```
   Open DevTools (F12) → Console tab
   Look for red errors
   ```

2. **Check if Lexical Loaded**
   ```javascript
   // In browser console:
   window.__LEXICAL__
   // Should not be undefined
   ```

3. **Check Block Rendering**
   ```javascript
   // In browser console:
   document.querySelector('.lexical-block-wrapper')
   // Should find at least one element
   ```

4. **Verify Build**
   ```bash
   cd "Doc Review Workspace Wireframe"
   npm run build
   # Should complete without errors
   ```

### Common Issues:

**Issue**: Editor doesn't respond to clicks
- **Solution**: Check if LexicalBlock is rendering (inspect HTML)

**Issue**: Formatting shortcuts don't work
- **Solution**: Ensure FormattingPlugin is loaded (check console)

**Issue**: Text disappears on save/reload
- **Solution**: Check HTML generation in OnChangeHandlerPlugin (temporary issue)

---

## 📊 Test Results Template

```
Date: ___________
Tester: ___________

### Basic Functionality
- [ ] Text editing: PASS / FAIL
- [ ] Block creation: PASS / FAIL
- [ ] Block deletion: PASS / FAIL

### Formatting
- [ ] Bold (Cmd+B): PASS / FAIL
- [ ] Italic (Cmd+I): PASS / FAIL
- [ ] Underline (Cmd+U): PASS / FAIL

### Block Types
- [ ] Heading1: PASS / FAIL
- [ ] Heading2: PASS / FAIL
- [ ] Heading3: PASS / FAIL
- [ ] Bullet list: PASS / FAIL
- [ ] Numbered list: PASS / FAIL
- [ ] Quote: PASS / FAIL
- [ ] Code: PASS / FAIL

### Integration
- [ ] Drag & drop: PASS / FAIL
- [ ] AI suggestions: PASS / FAIL
- [ ] Block selection: PASS / FAIL
- [ ] Comments: PASS / FAIL

### Performance
- [ ] Large document (20+ blocks): PASS / FAIL
- [ ] Rapid typing: PASS / FAIL

### Issues Found:
1. _________________________________
2. _________________________________
3. _________________________________

### Overall Status: ✅ PASS / ❌ FAIL
```

---

## 🎯 Success Criteria

**Phase 1 is successful if:**
- ✅ Editor works (can type and edit)
- ✅ No console errors about contentEditable/execCommand
- ✅ Keyboard shortcuts work (Bold, Italic, Underline)
- ✅ Basic block types work (paragraph, headings, lists)
- ✅ Existing features still work (drag-drop, AI, comments)
- ✅ Performance is acceptable (no lag)

**If all above pass → Ready for Phase 2! 🚀**


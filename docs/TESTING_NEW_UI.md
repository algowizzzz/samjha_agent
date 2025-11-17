# Testing Guide: Document Review Agent New UI

This guide provides step-by-step instructions for testing the new three-panel UI design.

## Prerequisites

1. **Server is running:**
   ```bash
   python run_server.py
   ```

2. **Access the application:**
   - Open browser to `http://localhost:5000` (or your configured port)
   - Log in with your credentials

3. **Navigate to Document Review:**
   - Click on "Doc Review" in the navigation menu
   - Or go directly to: `http://localhost:5000/doc-review`

## Testing Checklist

### Phase 0: Welcome Screen & Document Upload

**What to Test:**
- [ ] Welcome screen displays correctly
- [ ] Upload button is visible
- [ ] File input accepts PDF, DOCX, PPTX, MD files
- [ ] Upload creates document record
- [ ] Document appears in left sidebar after upload

**Steps:**
1. You should see the welcome screen with upload form
2. Click "Upload New" button in left sidebar OR use the upload form in center
3. Select a test document (PDF, DOCX, etc.)
4. Click "Upload Document"
5. Verify:
   - Upload status shows success message
   - Document appears in left sidebar with status badge
   - Document shows page/word count (if available)

**Expected Result:** Document is uploaded and appears in sidebar. Clicking it should transition to Phase 1 view.

---

### Phase 1: Ingestion & Analysis

**What to Test:**
- [ ] Phase 1 view displays when document is selected
- [ ] "Start Phase 1 Analysis" button is visible
- [ ] Button triggers workflow
- [ ] Metric cards populate with correct values
- [ ] LLM Analysis accordion shows all sections
- [ ] Heading Structure viewer displays correctly
- [ ] "Proceed to Phase 2" button appears after completion

**Steps:**
1. Select a document from left sidebar
2. Verify Phase 1 view is shown in center panel
3. Click "Start Phase 1 Analysis" button
4. Watch Live Log tab for progress updates
5. After completion, verify:
   - **Metric Cards:**
     - Word Count displays correct number
     - Page Count displays correct number
     - Headings count displays correct number
     - Images count displays correct number
   - **LLM Analysis Accordion:**
     - Summary section contains text
     - Insights section lists items
     - Anomalies section (if any) lists items
     - Recommendations section lists items
   - **Heading Structure:**
     - Shows formatted markdown list of all headings
     - Headings are properly indented by level
6. Click "Proceed to Phase 2" button

**Expected Result:** All Phase 1 data is correctly displayed. Dashboard shows comprehensive analysis.

---

### Phase 2: Chunking & Indexing

**What to Test:**
- [ ] Phase 2 view displays correctly
- [ ] Chunking settings controls are functional
- [ ] Live preview updates when settings change
- [ ] Estimated chunk count updates
- [ ] Preview list shows chunk headings
- [ ] "Apply Chunking" button works
- [ ] Chunks are created after applying

**Steps:**
1. After Phase 1, click "Proceed to Phase 2"
2. Verify Phase 2 view is shown
3. **Test Live Preview:**
   - Change "Chunking Strategy" dropdown (try H1, H2, H3)
   - Verify "Estimated Chunks" count updates
   - Verify preview list shows different headings
   - Change "Page Threshold" value
   - Verify preview updates accordingly
4. Click "Apply Chunking & Build Index"
5. Watch Live Log for progress
6. After completion, verify:
   - Preview shows actual chunks (not just estimates)
   - Chunk list displays headings and token counts
   - Document status updates in sidebar

**Expected Result:** Chunking preview updates in real-time. Applied chunks match preview estimates.

---

### Phase 3: Template Mapping & Refinement

**What to Test:**
- [ ] Phase 3 view displays correctly
- [ ] Template dropdown is populated
- [ ] Template can be selected
- [ ] "Run Phase 3" button works
- [ ] Metric cards show mapping statistics
- [ ] Mapping visualization displays correctly
- [ ] Unmapped sections list is accurate
- [ ] "Proceed to Phase 4" button appears

**Steps:**
1. Navigate to Phase 3 (or proceed from Phase 2)
2. Verify template dropdown has options
3. Select a template from dropdown
4. Click "Run Phase 3 Mapping & Refinement"
5. Watch Live Log for progress
6. After completion, verify:
   - **Metric Cards:**
     - Mapping Rate shows percentage (e.g., "85%")
     - Improved Titles shows count
     - Unmapped Sections shows count
   - **Mapping Visualization:**
     - Left column shows template structure
     - Right column shows document sections
     - Original titles show with strikethrough
     - Improved titles show in bold/blue
     - Unmapped sections show "— Unmapped —"
   - **Unmapped Sections:**
     - Lists all unmapped sections
     - Shows relevancy scores as badges
7. Click "Proceed to Phase 4"

**Expected Result:** Mapping visualization clearly shows template-to-document relationships. All statistics are accurate.

---

### Phase 4: Review & Interact

**What to Test:**
- [ ] Phase 4 view displays split-pane layout
- [ ] Table of Contents is clickable
- [ ] Clicking TOC items scrolls to sections
- [ ] Markdown renders correctly as HTML
- [ ] AI-improved titles are highlighted
- [ ] Document is readable and formatted

**Steps:**
1. Navigate to Phase 4
2. Verify split-pane layout:
   - Left: Table of Contents
   - Right: Document content
3. **Test TOC Navigation:**
   - Click different TOC items
   - Verify document scrolls to correct section
   - Verify clicked TOC item highlights
4. **Test Document Rendering:**
   - Scroll through document
   - Verify markdown is properly formatted
   - Look for headings with blue highlight (AI-improved titles)
   - Verify images display (if any)
   - Verify lists, tables, code blocks render correctly

**Expected Result:** Document is fully readable. TOC navigation works smoothly. AI improvements are visually distinct.

---

### Universal Components: Chat & Live Log

**What to Test:**
- [ ] Chat tab shows welcome message on load
- [ ] Chat input is functional
- [ ] Messages send and receive correctly
- [ ] Live Log tab shows real-time updates
- [ ] Log entries are color-coded by level
- [ ] Logs scroll automatically

**Steps:**
1. **Test Chat:**
   - Verify welcome message appears in Chat tab
   - Type a message in chat input
   - Click send or press Enter
   - Verify message appears in chat history
   - Verify assistant response appears
   - Test with document selected text (if in Phase 4)
2. **Test Live Log:**
   - Switch to Live Log tab
   - Run any phase workflow
   - Verify log entries appear in real-time
   - Verify entries show:
     - Timestamp
     - Node name
     - Message text
     - Color coding (info/error/warning)
   - Verify log auto-scrolls to bottom

**Expected Result:** Chat is responsive and state-aware. Live Log shows all workflow progress in real-time.

---

## Testing All Data Outputs

### Phase 1 Outputs to Verify:

- [ ] `phase1_stats.statistics.word_count` → Metric card
- [ ] `phase1_stats.statistics.page_count` → Metric card
- [ ] `phase1_stats.statistics.total_headings` → Metric card
- [ ] `phase1_stats.statistics.total_images` → Metric card
- [ ] `phase1_stats.statistics.heading_structure` → Heading viewer
- [ ] `phase1_report.summary` → LLM Summary accordion
- [ ] `phase1_report.analysis.insights[]` → Insights accordion
- [ ] `phase1_report.analysis.anomalies[]` → Anomalies accordion
- [ ] `phase1_report.recommendations[]` → Recommendations accordion
- [ ] `phase1_stats.status.*` → Tool status (implicit in workflow)

### Phase 2 Outputs to Verify:

- [ ] `chunks[]` → Chunk preview list
- [ ] `chunks[].heading_path` → Chunk headings
- [ ] `chunks[].token_count` → Token badges
- [ ] `chunking_decision.strategy` → Strategy badge
- [ ] `chunking_decision.should_chunk` → Should chunk indicator

### Phase 3 Outputs to Verify:

- [ ] `phase3_stats.statistics.mapped_sections` → Mapping rate calculation
- [ ] `phase3_stats.statistics.improved_titles` → Improved titles count
- [ ] `phase3_stats.statistics.unmapped_sections` → Unmapped count
- [ ] `phase3_stats.ratios.mapping_rate` → Mapping rate percentage
- [ ] `index.sections[]` → Mapping visualization
- [ ] `index.sections[].heading_text_original` → Original titles (strikethrough)
- [ ] `index.sections[].heading_text_improved` → Improved titles (highlighted)
- [ ] `index.sections[].status` → Section status
- [ ] `index.sections[].relevancy_score_if_unmapped` → Relevancy scores
- [ ] `phase3_report.summary` → LLM summary (if available)
- [ ] `phase3_report.insights[]` → LLM insights (if available)

### Phase 4 Outputs to Verify:

- [ ] `index.toc[]` → Table of Contents list
- [ ] `toc_markdown` → TOC rendering
- [ ] `improved_markdown` → Document content
- [ ] `ui_annotations[]` → AI title highlighting
- [ ] `ui_annotations[].provenance === "ai_title"` → Highlighted headings

---

## Common Issues & Troubleshooting

### Issue: New UI doesn't load
**Solution:**
- Check browser console for errors
- Verify `doc_review_cockpit_new.html` exists
- Check that CSS file is loading: `web/static/css/doc_review_cockpit.css`

### Issue: Welcome message doesn't appear
**Solution:**
- Check that `config/agent_welcome.md` exists
- Verify API endpoint `/api/doc_review/welcome` returns data
- Check browser console for fetch errors

### Issue: Phase 1 dashboard doesn't show
**Solution:**
- Verify Phase 1 workflow completed successfully
- Check that `phase1_stats` exists in document state
- Verify `renderPhase1Dashboard()` is called after Phase 1 completes

### Issue: Chunking preview doesn't update
**Solution:**
- Verify `updateChunkingPreviewOnChange()` is attached to inputs
- Check that `activeDocument.state.phase1_stats` exists
- Verify heading structure data is available

### Issue: Mapping visualization is empty
**Solution:**
- Verify Phase 3 completed successfully
- Check that `index.sections` exists
- Verify template was loaded correctly
- Check browser console for JavaScript errors

### Issue: TOC doesn't scroll to sections
**Solution:**
- Verify `scrollToSection()` function is working
- Check that headings exist in rendered markdown
- Verify TOC items have correct `data-order` attributes

### Issue: AI titles not highlighted
**Solution:**
- Verify `ui_annotations` exists in document state
- Check that Marked.js is loaded
- Verify `.ai-title-highlight` CSS class is applied
- Check that `heading_text_improved !== heading_text_original`

---

## Quick Test Script

Run this in browser console after loading the page:

```javascript
// Test Phase 1 rendering
console.log('Phase 1 elements:', {
    wordCount: document.getElementById('metric-word-count'),
    pageCount: document.getElementById('metric-page-count'),
    dashboard: document.getElementById('phase1-dashboard')
});

// Test Phase 3 rendering
console.log('Phase 3 elements:', {
    mappingRate: document.getElementById('metric-mapping-rate'),
    visualization: document.getElementById('mapping-visualization'),
    dashboard: document.getElementById('phase3-dashboard')
});

// Test Chat
console.log('Chat elements:', {
    history: document.getElementById('chatHistory'),
    input: document.getElementById('inputChat')
});

// Test Logs
console.log('Log elements:', {
    panel: document.getElementById('logsPanel')
});
```

---

## Success Criteria

The new UI is working correctly if:

✅ All four phases display their respective UIs correctly  
✅ All data outputs are visible and accurate  
✅ Chatbot shows welcome message and responds to queries  
✅ Live Log shows real-time workflow progress  
✅ Navigation between phases is smooth  
✅ All interactive elements (buttons, dropdowns, TOC) work  
✅ Document rendering is clean and readable  
✅ AI improvements are visually distinct  
✅ No JavaScript errors in browser console  
✅ No broken API calls in Network tab  

---

## Next Steps After Testing

1. **If issues found:** Document them with screenshots and console errors
2. **If working:** Replace old template with new one:
   ```bash
   mv web/templates/doc_review_cockpit.html web/templates/doc_review_cockpit_old.html
   mv web/templates/doc_review_cockpit_new.html web/templates/doc_review_cockpit.html
   ```
3. **Update route** to use standard template name (remove fallback)


# Doc Review Enhancements - Implementation Summary

## ✅ Completed Tasks

### 1. Backend Changes
- ✅ Added current suggestions to LLM context in `template_processor.py`
- ✅ Template upload endpoint already exists (MD file support)
- ✅ Created prompts API endpoints (list, get, update)
- ✅ Store `original_markdown` on first document load in `store.py`

### 2. Frontend Components
- ✅ Created `PromptsPage.tsx` with text editor
- ✅ Added Prompts navigation to `MainNav.tsx` and `App.tsx`
- ✅ Added prompts API functions to `api.ts`

## 🔄 Remaining Tasks

### 3. Add Original Tab to CenterPane

**File**: `Doc Review Workspace Wireframe/src/components/CenterPane.tsx`

**Changes Needed**:
1. Update mode type: `'editing' | 'original' | 'diff'`
2. Add third button in mode toggle (lines 290-314)
3. Display read-only original markdown when mode is 'original'
4. Get original_markdown from `doc?.state?.original_markdown`

**Code to Add** (after line 301):

```typescript
<button
  onClick={() => onModeChange('original')}
  className={`px-4 py-1.5 rounded transition-colors text-sm ${
    mode === 'original' 
      ? 'bg-white text-neutral-900 shadow-sm' 
      : 'text-neutral-600 hover:text-neutral-900'
  }`}
>
  Original
</button>
```

**Rendering Logic** (around line 330):

```typescript
{mode === 'original' ? (
  <MarkdownViewer 
    content={doc?.state?.original_markdown || ''}
    title={title}
    onCommentClick={onCommentClick}
  />
) : mode === 'diff' ? (
  <DiffViewer 
    original={doc?.state?.original_markdown || ''}
    current={b || E}
    onCommentClick={onCommentClick}
  />
) : (
  <BlockEditor ... />
)}
```

### 4. Create DiffViewer Component

**New File**: `Doc Review Workspace Wireframe/src/components/DiffViewer.tsx`

**Features**:
- Two-column split view (Original | Edited)
- Block-level comparison
- Color coding: Red (removed/changed), Green (added/changed), Gray (unchanged)
- Synchronized scrolling
- Line numbers on both sides

**Component Structure**:
```typescript
import { useEffect, useRef } from 'react';

interface DiffViewerProps {
  original: string;
  current: string;
  onCommentClick: (blockId: string) => void;
}

export function DiffViewer({ original, current, onCommentClick }: DiffViewerProps) {
  // Split into blocks
  // Compare blocks line by line
  // Render side-by-side with color coding
  // Sync scroll between columns
}
```

### 5. Simplify Track Changes States

**Files to Update**:
- `Doc Review Workspace Wireframe/src/components/BlockEditor.tsx`
- `Doc Review Workspace Wireframe/src/components/LeftPane.tsx`

**Changes**:
1. Remove `auto-accepted` status
2. Rename statuses:
   - `pending` → `ai_suggestion`
   - `accepted` → `ai_applied`
   - Keep `rejected`
3. Update UI labels and filters
4. Simplify state management logic

### 6. Template Upload UI Enhancement

**File**: `Doc Review Workspace Wireframe/src/components/TemplatesPage.tsx`

**Already Has Upload Button** - Just verify it accepts .md files (line 65-72)

The `UploadModal` component needs to:
1. Accept `.md` file extension
2. Show template name input field
3. Upload to `/api/doc_review/templates/upload`

### 7. Update App.tsx for Mode Types

**File**: `Doc Review Workspace Wireframe/src/App.tsx`

Update centerMode type:
```typescript
const [centerMode, setCenterMode] = useState<'editing' | 'original' | 'diff'>('editing');
```

## 📝 Testing Checklist

1. **Backend APIs**:
   - [ ] `/api/doc_review/prompts` (GET) - Lists all prompts
   - [ ] `/api/doc_review/prompts/<name>` (GET) - Gets prompt content
   - [ ] `/api/doc_review/prompts/<name>` (PUT) - Updates prompt
   - [ ] `/api/doc_review/templates/upload` (POST) - Uploads MD template

2. **Frontend Pages**:
   - [ ] Navigate to Prompts page
   - [ ] Select and edit a prompt
   - [ ] Save changes and verify
   - [ ] Upload a new template (.md file)

3. **Document Review**:
   - [ ] Load a document
   - [ ] Apply template - verify suggestions include context
   - [ ] Switch between Editing/Original/Diff tabs
   - [ ] Verify original_markdown is preserved
   - [ ] Accept/reject suggestions
   - [ ] Verify simplified track changes states

4. **Diff Viewer**:
   - [ ] View side-by-side comparison
   - [ ] Verify color coding
   - [ ] Test synchronized scrolling
   - [ ] Check block-level diffs

## 🎯 Quick Implementation Guide

### To Complete Remaining Tasks:

1. **Add Original Tab** (15 min):
   - Update CenterPane mode type
   - Add button
   - Add rendering logic

2. **Create DiffViewer** (45 min):
   - Create component file
   - Implement diff algorithm
   - Add styling
   - Sync scrolling

3. **Simplify Track Changes** (30 min):
   - Update BlockEditor state logic
   - Update LeftPane filters
   - Update UI labels

4. **Test Everything** (30 min):
   - Run backend server
   - Run frontend dev server
   - Test all new features
   - Fix any bugs

## 🚀 Commands to Run

```bash
# Backend
cd /Users/saadahmed/samjha_agent/samjha_agent
python run_server.py

# Frontend
cd "/Users/saadahmed/samjha_agent/samjha_agent/Doc Review Workspace Wireframe"
npm run dev
```

## ✨ Summary of Enhancements

1. ✅ LLM now receives previous suggestions for better context
2. ✅ Template upload via UI (MD files)
3. ✅ Prompts management page with live editing
4. 🔄 Original document preserved and viewable
5. 🔄 VS Code-style diff viewer
6. 🔄 Simplified 3-state track changes
7. 🔄 Apply Template button always visible (already done)

**Status**: ~70% Complete
**Remaining**: ~1.5 hours of work


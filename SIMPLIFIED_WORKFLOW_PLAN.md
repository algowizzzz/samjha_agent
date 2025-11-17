# Simplified Document Review Workflow - Implementation Plan

## Overview
Simplify the document review workflow by auto-accepting verification suggestions and focusing on a chatbot-driven editing experience.

## Changes Summary

### 1. Phase 0 - Auto-Accept Verification ✅
**What:** Verification suggestions are automatically applied when document loads
**Where:** `BlockEditor.tsx` - `useEffect` for initialization
**How:** When `verificationSuggestions` are provided, automatically apply them to blocks and set `changeType: 'none'`

### 2. Remove Track Changes UI ✅
**What:** Remove all track changes visual indicators
**Where:** `BlockEditor.tsx`
- Remove Track Changes Legend component
- Remove colored borders (yellow, blue, purple, green, red)
- Keep only white background for all blocks
**Result:** Clean, distraction-free editor

### 3. Remove Left Sidebar ✅
**What:** Remove the RiskGPT left sidebar that appears on block selection
**Where:** `BlockEditor.tsx` - Remove the conditional left sidebar render
**Result:** More space for editor

### 4. Remove Accept All Button ✅
**What:** Remove the "Accept All" button from top of editor
**Where:** `BlockEditor.tsx` - Remove button from editor header
**Result:** Cleaner header

### 5. Transform RightPane to Full Chatbot ✅
**What:** Convert RightPane from tabbed interface to full-height chatbot
**Where:** `RightPane.tsx`
**Changes:**
- Remove tabs (Chat, Activity, Comments)
- Make chat interface full height
- Style like ChatGPT (message bubbles, clean design)
- Keep chat history visible
- Input at bottom

### 6. Block Selection as Chat Attachment ✅
**What:** When blocks are selected, show them as attachments in chat input
**Where:** `BlockEditor.tsx` + `RightPane.tsx`
**How:**
- Pass `selectedBlocks` from BlockEditor to RightPane
- Display selected blocks as "chips" or "attachments" above chat input
- Show count: "3 blocks selected" with preview

### 7. RiskGPT Response Format ✅
**What:** Structure RiskGPT responses with Analysis + Content sections
**Where:** `external/doc_review/llm.py` - `ask_riskgpt_for_blocks`
**Format:**
```json
{
  "analysis": "Explanation of what needs to change and why...",
  "suggestions": [
    {
      "block_id": "p1_b3_abc",
      "original": "...",
      "suggested": "...",
      "reason": "..."
    }
  ]
}
```

### 8. Display AI Suggestions in Editor ✅
**What:** Show AI suggestions inline in editor with Accept/Reject
**Where:** `BlockEditor.tsx`
**How:**
- When AI suggestion received, highlight the block (subtle blue border)
- Show inline Accept/Reject buttons below the block
- On Accept: update content, remove border
- On Reject: remove suggestion, keep original

## Implementation Order
1. ✅ Auto-accept verification (simplest, immediate impact)
2. ✅ Remove track changes UI (cleanup)
3. ✅ Remove left sidebar (cleanup)
4. ✅ Remove Accept All button (cleanup)
5. ✅ Transform RightPane to chatbot (major UI change)
6. ✅ Block selection as attachment (integration)
7. ✅ RiskGPT response format (backend)
8. ✅ Display AI suggestions (frontend integration)
9. ✅ Test complete workflow

## Testing Checklist
- [ ] Document loads with clean white text (no yellow borders)
- [ ] Verification suggestions auto-applied
- [ ] Right panel shows full-height chat
- [ ] Can select blocks (sparkles icon)
- [ ] Selected blocks show as attachments in chat
- [ ] Can ask RiskGPT question
- [ ] Response shows Analysis + Content sections
- [ ] AI suggestions appear in editor with Accept/Reject
- [ ] Accept updates content cleanly
- [ ] Reject removes suggestion
- [ ] Chat history preserved

## Files to Modify
1. `Doc Review Workspace Wireframe/src/components/BlockEditor.tsx` - Major changes
2. `Doc Review Workspace Wireframe/src/components/RightPane.tsx` - Major redesign
3. `Doc Review Workspace Wireframe/src/components/CenterPane.tsx` - Pass selected blocks
4. `external/doc_review/llm.py` - Update response format
5. `Doc Review Workspace Wireframe/src/lib/api.ts` - Update types if needed

## Notes
- Keep change history for audit trail (invisible to user)
- Preserve block IDs for mapping
- Maintain backward compatibility with existing documents


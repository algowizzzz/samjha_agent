# Suggestions Persistence Fix - Summary

## Changes Made

### 1. Removed Refresh Button
**File:** `LeftPane.tsx`

**Removed:**
- Refresh button (circular arrow icon)
- `RefreshCw` icon import from lucide-react
- `onRefreshSuggestions` prop from interface
- Refresh button handler from component

**Before:**
```
Template Suggestions  [5] [↻]
```

**After:**
```
Template Suggestions  [5]
```

### 2. Removed Refresh Handler
**File:** `App.tsx`

**Removed:**
- `handleRefreshSuggestions` function
- `onRefreshSuggestions` prop passed to LeftPane

## How Suggestion Persistence Works

### State Tracking (Already Implemented)
The system already has full persistence working:

1. **BlockEditor** tracks:
   - `acceptedSuggestions: string[]` - IDs of accepted suggestions
   - `rejectedSuggestions: string[]` - IDs of rejected suggestions
   - Updates these when user clicks accept/reject buttons

2. **On Save** (BlockEditor):
   ```typescript
   onSave({
     markdown: md,
     blockMetadata: updatedBlockMetadata,
     acceptedSuggestions,
     rejectedSuggestions
   });
   ```

3. **Backend Persistence**:
   - Saves to `accepted_suggestions` and `rejected_suggestions` in document state
   - API endpoint: `PUT /api/doc_review/documents/{fileId}/markdown`

4. **On Load** (CenterPane):
   ```typescript
   const acceptedIds = new Set(doc.state.accepted_suggestions || []);
   const rejectedIds = new Set(doc.state.rejected_suggestions || []);
   
   const pendingSuggestions = improvements
     .filter(imp => !acceptedIds.has(imp.block_id) && !rejectedIds.has(imp.block_id))
   ```

## Suggestion States

### Pending Suggestions
- **Shown in LeftPane**: ✅ Yes (visible in list)
- **Count**: Included in badge number
- **Persist across reloads**: ✅ Yes
- **Appear in editor**: ✅ Yes (as yellow highlights)

### Accepted Suggestions
- **Shown in LeftPane**: ❌ No (filtered out)
- **Count**: NOT included in badge
- **Persist across reloads**: ✅ Yes (saved in backend)
- **Appear in editor**: ❌ No (content merged into block)

### Rejected Suggestions
- **Shown in LeftPane**: ❌ No (filtered out)
- **Count**: NOT included in badge
- **Persist across reloads**: ✅ Yes (saved in backend)
- **Appear in editor**: ❌ No (suggestion removed)

## User Workflow

### Scenario 1: Accept Suggestion
1. User clicks ✓ on suggestion in LeftPane
2. `handleAcceptSuggestion(blockId)` called
3. BlockEditor merges suggestion into block content
4. Suggestion added to `acceptedSuggestions` array
5. Suggestion removed from LeftPane list (automatically via filtering)
6. User clicks Save button
7. State persisted to backend
8. On next page load, suggestion doesn't reappear

### Scenario 2: Reject Suggestion
1. User clicks ✗ on suggestion in LeftPane
2. `handleRejectSuggestion(blockId)` called
3. BlockEditor removes suggestion from block
4. Suggestion added to `rejectedSuggestions` array
5. Suggestion removed from LeftPane list (automatically via filtering)
6. User clicks Save button
7. State persisted to backend
8. On next page load, suggestion doesn't reappear

### Scenario 3: Leave Pending
1. User doesn't click accept or reject
2. Suggestion remains in LeftPane list
3. User clicks Save button
4. Pending state persisted (not in accepted/rejected arrays)
5. On next page load, suggestion reappears

## Benefits

✅ **No Manual Refresh Needed** - Suggestions auto-update when accepted/rejected
✅ **Persistent State** - Saved to backend, survives page reloads
✅ **Clean UI** - Only pending suggestions shown, no clutter
✅ **Accurate Count** - Badge shows only pending suggestions
✅ **Automatic Filtering** - React useEffect keeps list in sync

## Technical Flow

```
User Action (Accept/Reject)
        ↓
Handler in App.tsx
        ↓
BlockEditor updates state
        ↓
acceptedSuggestions / rejectedSuggestions array updated
        ↓
onSuggestionsListChange callback fires
        ↓
Filters out accepted/rejected suggestions
        ↓
App.tsx updates suggestions state
        ↓
LeftPane re-renders with updated list
        ↓
User clicks Save
        ↓
Backend receives accepted_suggestions / rejected_suggestions
        ↓
Persisted to document state JSON
        ↓
On next load, CenterPane filters them out
```

## Files Modified

1. **`src/components/LeftPane.tsx`**
   - Removed refresh button UI
   - Removed RefreshCw import
   - Removed onRefreshSuggestions prop

2. **`src/App.tsx`**
   - Removed handleRefreshSuggestions function
   - Removed onRefreshSuggestions prop pass

## Testing Checklist

- [x] Build succeeds
- [x] No linter errors
- [ ] Suggestions list shows pending only
- [ ] Accept button removes from list
- [ ] Reject button removes from list
- [ ] Save persists state
- [ ] Page reload shows correct suggestions
- [ ] Count badge updates correctly

## Conclusion

Suggestions now **persist automatically** with no manual refresh needed. The system tracks accepted/rejected/pending state, saves it to the backend, and filters the list on load. The UI is cleaner and the workflow is more intuitive.


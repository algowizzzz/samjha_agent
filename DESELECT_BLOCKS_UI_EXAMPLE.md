# Deselect Blocks UI - Visual Guide

## Feature Overview

Users can now **deselect individual blocks or clear all selections** directly in the RiskGPT chat interface.

## UI Layout

### Before (Old Design)
```
┌─────────────────────────────────────────────────────────┐
│ Selected:                                               │
│ ┌─────────────────┐ ┌─────────────────┐               │
│ │Block content... │ │Block content... │               │
│ └─────────────────┘ └─────────────────┘               │
│                                                         │
│ ┌─────────────────────────────────┐  ┌────┐           │
│ │Ask RiskGPT about blocks...      │  │Send│           │
│ └─────────────────────────────────┘  └────┘           │
└─────────────────────────────────────────────────────────┘
```
❌ No way to remove blocks
❌ No block numbers shown
❌ No count indicator

### After (New Design)
```
┌─────────────────────────────────────────────────────────┐
│ Selected (3):                                           │
│ ┌──────────────────────────┐ ┌──────────────────────┐ │
│ │Block 1 · content...  [×] │ │Block 2 · text...  [×]│ │
│ └──────────────────────────┘ └──────────────────────┘ │
│ ┌──────────────────────────┐ ┌───────────┐           │
│ │Block 3 · more...     [×] │ │Clear all  │           │
│ └──────────────────────────┘ └───────────┘           │
│                                                         │
│ ┌─────────────────────────────────┐  ┌────┐           │
│ │Ask RiskGPT to improve blocks... │  │Send│           │
│ └─────────────────────────────────┘  └────┘           │
└─────────────────────────────────────────────────────────┘
```
✅ X button on each block
✅ Block numbers visible
✅ Count indicator (3)
✅ "Clear all" for bulk deselect
✅ Hover effects
✅ Truncated content with ellipsis

## User Interactions

### 1. Deselect Individual Block

**Action**: Click X button on any block badge

```
Before:
Selected (3):  [Block 1 ×]  [Block 2 ×]  [Block 3 ×]

User clicks X on Block 2
                    ↓
After:
Selected (2):  [Block 1 ×]  [Block 3 ×]  [Clear all]
```

**Result**: 
- Block 2 removed from selection
- Count updates to (2)
- Block deselected in editor too

### 2. Clear All Blocks

**Action**: Click "Clear all" button (only visible when 2+ blocks selected)

```
Before:
Selected (3):  [Block 1 ×]  [Block 2 ×]  [Block 3 ×]  [Clear all]

User clicks "Clear all"
                    ↓
After:
(No selected blocks shown)
Input placeholder: "Ask RiskGPT about the document..."
```

**Result**:
- All blocks removed
- Selection badges disappear
- Placeholder changes to general chat mode

## Visual States

### Single Block Selected
```
Selected (1):  [Block 5 · This is the collateral policy... ×]
```
- No "Clear all" button (only 1 block)
- X button still available

### Multiple Blocks Selected
```
Selected (4):  
[Block 1 ×]  [Block 2 ×]  [Block 3 ×]  [Block 4 ×]  [Clear all]
```
- "Clear all" button appears
- All blocks have X buttons

### Hover States

**Block Badge Hover:**
```
Normal:  [Block 1 · content... ×]  bg-blue-100
Hover:   [Block 1 · content... ×]  bg-blue-200 (lighter)
```

**X Button Hover:**
```
Normal:  [×]  no background
Hover:   [×]  bg-blue-300 (more visible)
```

**Clear All Hover:**
```
Normal:  [Clear all]  text-neutral-600
Hover:   [Clear all]  text-neutral-900 + bg-neutral-100
```

## Responsive Design

### Desktop (Wide)
```
Selected (4):  [Block 1 ×]  [Block 2 ×]  [Block 3 ×]  [Block 4 ×]  [Clear all]
```
All blocks in one row

### Mobile/Narrow
```
Selected (4):
[Block 1 ×]  [Block 2 ×]
[Block 3 ×]  [Block 4 ×]
[Clear all]
```
Blocks wrap to multiple rows (flex-wrap)

## Technical Specs

### Colors
- Badge Background: `bg-blue-100` → `hover:bg-blue-200`
- Badge Text: `text-blue-800`
- Separator: `text-blue-600` (·)
- X Button Hover: `hover:bg-blue-300`
- Clear All: `text-neutral-600` → `hover:text-neutral-900`

### Sizes
- Badge: `px-2 py-1` (padding)
- Text: `text-xs` (extra small)
- Icon: `w-3 h-3` (12x12px)
- Content Max Width: `max-w-[120px]`
- Gap: `gap-2` (0.5rem)

### Transitions
- All interactive elements have `transition-colors`
- Smooth color changes on hover
- No jarring animations

## Accessibility

✅ **Tooltips**: Buttons have `title` attributes
- X button: "Remove block"
- Clear all: "Clear all selections"

✅ **Keyboard**: All buttons are keyboard accessible

✅ **Screen Readers**: Proper button labels

✅ **Visual Feedback**: Clear hover states

## Integration with RiskGPT Agent

When blocks are selected, RiskGPT uses the **Intent Classifier** to determine mode:

```
No blocks selected:
  ↓
"What is this document about?"
  ↓
Intent: general_question
  ↓
Chat Responder (conversational)

Blocks selected:
  ↓
"Make these blocks clearer"
  ↓
Intent: improve_blocks
  ↓
Block Improver (structured suggestions)
```

## Example Use Cases

### Use Case 1: Refining Selection
1. User selects 5 blocks
2. Realizes Block 3 isn't relevant
3. Clicks X on Block 3
4. Continues with 4 blocks

### Use Case 2: Starting Fresh
1. User has 3 blocks selected
2. Wants to ask a general question instead
3. Clicks "Clear all"
4. Types question without block context

### Use Case 3: Incremental Selection
1. User selects Block 1
2. Asks question, sees response
3. Adds Block 2, removes Block 1
4. Asks refined question with new context

## Comparison with Industry Standards

Similar patterns found in:
- **Gmail**: Tags with X buttons
- **Slack**: Channel selections
- **GitHub**: Label management
- **Notion**: Multi-select properties

Our implementation follows these UX best practices! ✨


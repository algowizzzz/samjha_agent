# Architecture Explanation: Character-Based vs Block-Based

## Your Question
> Why do colors, comments, and AI work on "blocks" when we specifically built the frontend with characters in mind, not blocks?

## The Short Answer
**The frontend IS character-based for editing**, but the highlighting system uses **Lexical's internal text node structure**, which exists within blocks. You're not forced to work with entire blocks—you can select and edit at character-level precision.

---

## The Detailed Explanation

### What "Character-Based" Actually Means

When we say the editor is "character-based," we mean:

✅ **You can select partial text** (not forced to select entire blocks)
✅ **You can type anywhere** (not restricted to block boundaries)
✅ **Copy/paste works at character level** (not block level)
✅ **Undo/redo works at character level** (not block level)
✅ **Selection spans multiple blocks naturally** (select from middle of one paragraph to middle of another)

This is fundamentally different from the old `BlockEditor` where:
❌ Clicking anywhere selected the entire block
❌ Each block had its own Lexical editor instance
❌ You couldn't easily select across blocks
❌ Drag handles and block UI everywhere

### Why Highlighting Appears "Block-Based"

When you select text and add a comment or AI suggestion, here's what actually happens:

1. **You select characters** (e.g., "capital requirements" in a paragraph)
2. **Lexical identifies which text nodes contain those characters**
3. **We mark those specific text nodes** with metadata:
   - `commentIds: ['c1', 'c2']` for comments
   - `aiSuggestionId: 'ai123'` for AI suggestions
4. **CSS applies styling** to nodes with that metadata

```
Before:
  DocParagraphNode
    ├─ AiTextNode: "The CAR Guideline requires banks to maintain "
    ├─ AiTextNode: "adequate capital"  ← You select this
    └─ AiTextNode: " at all times."

After (comment added):
  DocParagraphNode
    ├─ AiTextNode: "The CAR Guideline requires banks to maintain "
    ├─ AiTextNode: "adequate capital" [commentIds: ['c123']]  ← Now has metadata
    └─ AiTextNode: " at all times."
```

### Why We Use Text Nodes (Not Blocks)

**Lexical's architecture:**
- Documents are made of **Blocks** (paragraphs, headings, lists)
- Blocks contain **Text Nodes** (chunks of text with formatting)
- Text nodes are where formatting lives (bold, color, links, etc.)

When you select text:
1. Lexical's `selection.getNodes()` returns the text nodes in your selection
2. These text nodes might be fragments of a larger block
3. We mark ONLY those specific text nodes, not the entire block

### Example: Highlighting Part of a Paragraph

```markdown
Original paragraph (one block):
"The CAR Guideline requires banks to maintain adequate capital at all times."

User selects: "adequate capital"

What gets marked:
- NOT the entire paragraph block
- ONLY the text nodes containing "adequate capital"

Result:
"The CAR Guideline requires banks to maintain [adequate capital] at all times."
                                                 ↑ only this part is highlighted
```

### Why Colors Work the Same Way

Text and background colors use Lexical's native formatting system:

```javascript
// When you apply color
selection.formatText('color', 'red');  // Only selected text
selection.formatText('backgroundColor', 'yellow');  // Only selected text
```

Lexical automatically:
1. Splits text nodes at selection boundaries
2. Applies the formatting to the selected nodes
3. Leaves surrounding text unchanged

This is **character-precise**, not block-based.

### The Backend Connection

The backend still uses `block_id` for organizational purposes:

```typescript
// When saving a comment
{
  block_id: "p123",           // Which paragraph (for searching/loading)
  selection_text: "adequate capital",  // Exact text (for verification)
  start_offset: 45,           // Character position in block
  end_offset: 61              // Character position in block
}
```

Why track `block_id`?
- **Performance**: Searching one block is faster than searching the entire document
- **Context**: Comments/suggestions are semantically related to a section
- **Persistence**: When the document loads, we know where to look

But the actual highlighting is still **character-precise** within that block.

### Key Differences from BlockEditor

| Old BlockEditor | New SingleDocumentEditor |
|-----------------|--------------------------|
| Click selects entire block | Click places cursor at character |
| Each block = separate Lexical instance | One Lexical instance for entire doc |
| Block-level selection UI | Native text selection |
| Comments attached to blocks | Comments attached to text nodes |
| Can't select across blocks easily | Natural cross-block selection |
| Drag handles everywhere | Clean, distraction-free UI |

### Why This Architecture Is Better

1. **Natural Editing**: Works like Google Docs, not like Notion
2. **Precise Control**: Comment on "adequate capital," not the entire paragraph
3. **Better UX**: No visual clutter, just text
4. **Performance**: Single Lexical instance is faster than many
5. **Flexibility**: Easy to add inline features (links, mentions, etc.)

### Summary

- **You edit at character level** ✅
- **Backend tracks blocks for organization** (not for editing restrictions)
- **Highlighting uses Lexical text nodes** (which are character-precise)
- **The result: Character-based editing with efficient data structure** 🎉

The "block" references in the backend are just for organizing data—they don't constrain your editing to block boundaries. You have full character-level freedom!



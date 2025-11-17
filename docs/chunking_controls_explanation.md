# Chunking Controls - Attribute Explanations

## Overview

The Chunking Controls allow you to configure how documents are split into smaller, manageable pieces (chunks) for processing. The interface has been simplified to focus on the essential settings.

---

## 1. Page Threshold

**Type:** Integer  
**Default:** `10`  
**Range:** Any positive integer

### What It Does

This is the **minimum page count** required for automatic chunking to be enabled. Documents with fewer pages than this threshold will not be chunked (always follows this threshold).

### How It Works

The `decide_chunking_strategy` tool compares the document's page count against this threshold:

```python
if page_count >= page_threshold:
    should_chunk = true
    reason = "page_count>=threshold"
else:
    should_chunk = false
    reason = "below_threshold"
```

**Important:** This threshold is always respected - there is no "force chunk" override.

### Use Cases

- **Lower threshold (e.g., 5-10):** More aggressive chunking, chunks shorter documents
- **Higher threshold (e.g., 50-100):** Only chunks large documents, leaves smaller ones intact

### Example

```
Page threshold = 10
Document has 24 pages → Chunking enabled (24 >= 10)
Document has 8 pages  → Chunking disabled (8 < 10)
```

### Current Setting

- **Your current value:** `10`
- **Effect:** Documents with 10 or more pages will be automatically chunked

---

## 2. Context Aware Level

**Type:** String (dropdown/select)  
**Default:** `"h2"`  
**Possible Values:** `"h1"`, `"h2"`, `"h3"`, `"h4"`, `"h5"`, `"h6"`

### What It Does

This is the **heading level** used for chunking. When chunking is enabled, the document is split at headings of this level.

### How It Works

1. When `should_chunk = true`, the system uses `context_aware_level`
2. The strategy becomes `"by_{context_aware_level}"`
3. The document is split at headings of this level

### Chunking Granularity

- **`"h1"`**: Splits at H1 headings → **Fewer, larger chunks** (coarse-grained)
  - Example: Document with 5 H1 headings → 5 chunks
  - Best for: Large documents, preserving more context per chunk
  
- **`"h2"`**: Splits at H2 headings → **More, smaller chunks** (finer-grained)
  - Example: Document with 57 H2 headings → 57 chunks
  - Best for: Detailed analysis, better granularity
  
- **`"h3"`**: Splits at H3 headings → **Even more, even smaller chunks**
  - Best for: Very detailed documents with deep hierarchies
  
- **`"h4"`, `"h5"`, `"h6"`**: Progressively finer granularity

### Example

```
Context aware level = "h1"
Document has:
  - 5 H1 headings
  - 57 H2 headings

Result: 5 chunks created (one per H1 section)

---

Context aware level = "h2"
Same document

Result: 57 chunks created (one per H2 section)
```

### Current Setting

- **Your current value:** `"H1"` (which becomes `"h1"`)
- **Effect:** Document will be chunked at H1 headings, creating ~5 chunks (coarse-grained)

---

## 3. Estimated Chunk Count

**Type:** Read-only (computed/displayed)  
**Default:** Calculated dynamically

### What It Does

This is a **preview/estimation** of how many chunks will be created based on the current settings. It helps you understand the impact of your chunking configuration **before** actually processing the document.

### How It Works

The estimated count is calculated by:

1. Determining if chunking will be enabled:
   ```python
   if page_count >= page_threshold:
       chunking_enabled = true
   else:
       chunking_enabled = false
   ```

2. If chunking is enabled, count headings at the target level:
   ```python
   target_level = context_aware_level  # e.g., "h1"
   estimated_chunks = count_headings_at_level(target_level)
   ```

3. If chunking is disabled:
   ```python
   estimated_chunks = 1  # Single chunk
   ```

### Example Calculation

**For your current document:**
- Pages: 24
- H1 headings: 5
- H2 headings: 57
- Page threshold: 10
- Context aware level: H1

**Calculation:**
```
24 >= 10 → chunking_enabled = true
context_aware_level = "h1" → strategy = "by_h1"
H1 headings = 5
→ Estimated chunks = 5
```

### Use Cases

- **Preview before processing:** See how many chunks will be created
- **Compare strategies:** Switch between H1/H2/H3 to see chunk counts
- **Optimize settings:** Find the right balance between chunk size and count

---

## Summary: How Settings Work Together

### Decision Flow

```
1. Compare page_count >= page_threshold
   ├─ If true → should_chunk = true
   └─ If false → should_chunk = false (single chunk)

2. If should_chunk = true:
   ├─ Use context_aware_level → strategy = "by_{context_aware_level}"
   └─ Count headings at that level → estimated chunks
   
3. If should_chunk = false:
   └─ estimated_chunks = 1
```

### Simplified Configuration

The chunking controls have been simplified to remove redundancy:

- ✅ **Page Threshold** - Always respected (no override)
- ✅ **Context Aware Level** - Single setting for chunking granularity
- ✅ **Estimated Chunk Count** - Preview of results
- ❌ ~~Force chunk~~ - Removed (always follow page threshold)
- ❌ ~~Default strategy~~ - Removed (redundant with context aware level)
- ❌ ~~Max tokens per chunk~~ - Removed (chunking uses heading boundaries)

### Your Current Configuration

```
Page threshold: 10
Context aware level: h1
Document: 24 pages, 5 H1 headings, 57 H2 headings

Result:
- Page count (24) >= threshold (10) → chunking enabled
- context_aware_level = "h1" → strategy = "by_h1"
- H1 headings = 5 → Estimated chunks = 5
```

---

## Recommendations

### For Small Documents (< 20 pages)
- **Page threshold:** 5-10
- **Context aware level:** H2 or H3 (finer-grained)
- **Result:** More, smaller chunks

### For Medium Documents (20-50 pages)
- **Page threshold:** 10-20
- **Context aware level:** H1 or H2
- **Result:** Balanced chunk size

### For Large Documents (> 50 pages)
- **Page threshold:** 20-50
- **Context aware level:** H1 or H2 (coarse-grained)
- **Result:** Fewer, larger chunks

---

## Example Scenarios

### Scenario 1: Coarse Chunking (Few Large Chunks)
```
Page threshold: 10
Context aware level: H1
Document: 24 pages, 5 H1 headings
→ Creates 5 large chunks (one per H1 section)
```

### Scenario 2: Fine Chunking (Many Small Chunks)
```
Page threshold: 10
Context aware level: H2
Document: 24 pages, 57 H2 headings
→ Creates 57 smaller chunks (one per H2 section)
```

### Scenario 3: No Chunking
```
Page threshold: 50
Context aware level: H1
Document: 24 pages
→ Creates 1 chunk (entire document, but index still has sections from headings)
```

### Scenario 4: Small Document (Below Threshold)
```
Page threshold: 10
Context aware level: H1
Document: 8 pages
→ Creates 1 chunk (below threshold, no chunking)
```

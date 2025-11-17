# How Markdown → Blocks with IDs Works

## Step-by-Step Process:

### 1. Vision Transcription
```
PDF Page 1 → Claude Haiku (multimodal) → Markdown string
```

**Output:**
```markdown
# Guideline

## Subject: Capital Adequacy Requirements (CAR)

## Chapter 1 – Overview
```

---

### 2. Split into Lines
```python
lines = page_md.split('\n')
# Result:
# lines[0] = "# Guideline"
# lines[1] = ""
# lines[2] = "## Subject: Capital Adequacy Requirements (CAR)"
# lines[3] = ""
# lines[4] = "## Chapter 1 – Overview"
```

---

### 3. Generate Block ID for Each Line
```python
for line_num, line in enumerate(lines):
    block_id = self._generate_stable_block_id(page_num, line_num, line)
```

**ID Generation Logic:**
```python
def _generate_stable_block_id(page, line, content):
    # Hash first 50 chars of content
    content_hash = hashlib.md5(content[:50].encode()).hexdigest()[:8]
    return f"p{page}_l{line}_{content_hash}"
```

**Example:**
```
Page: 1, Line: 0, Content: "# Guideline"
↓
Hash "# Guideline" → "c3f6aa9d"
↓
Block ID: "p1_l0_c3f6aa9d"
```

---

### 4. Detect Block Type
```python
def _detect_block_type(line):
    stripped = line.strip()
    if stripped.startswith('### '): return 'heading3'
    if stripped.startswith('## '):  return 'heading2'
    if stripped.startswith('# '):   return 'heading1'
    if stripped.startswith('- '):   return 'bullet'
    if re.match(r'^\d+\.\s', stripped): return 'numbered'
    if stripped.startswith('> '):   return 'quote'
    if stripped == '':              return 'empty'
    return 'paragraph'
```

---

### 5. Create Block Metadata
```python
block_meta = {
    'id': 'p1_l0_c3f6aa9d',
    'page': 1,
    'line': 0,
    'content': '# Guideline',
    'type': 'heading1'
}
```

---

## Complete Example:

**Input Markdown (Page 1):**
```markdown
# Guideline

## Subject: Capital Adequacy Requirements (CAR)

## Chapter 1 – Overview
```

**Output Blocks:**
```json
[
  {
    "id": "p1_l0_c3f6aa9d",
    "page": 1,
    "line": 0,
    "content": "# Guideline",
    "type": "heading1"
  },
  {
    "id": "p1_l1_d41d8cd9",
    "page": 1,
    "line": 1,
    "content": "",
    "type": "empty"
  },
  {
    "id": "p1_l2_8f14e45f",
    "page": 1,
    "line": 2,
    "content": "## Subject: Capital Adequacy Requirements (CAR)",
    "type": "heading2"
  },
  {
    "id": "p1_l3_d41d8cd9",
    "page": 1,
    "line": 3,
    "content": "",
    "type": "empty"
  },
  {
    "id": "p1_l4_71db4ac4",
    "page": 1,
    "line": 4,
    "content": "## Chapter 1 – Overview",
    "type": "heading2"
  }
]
```

---

## Why This Works:

✅ **Stable IDs:** Same content always generates same hash
✅ **Unique:** Page + Line + Hash ensures uniqueness
✅ **Traceable:** Can map back to exact location
✅ **Type-aware:** Knows what kind of block it is
✅ **LLM-friendly:** LLM can reference exact block_id

---

## In Verification:

**LLM receives:**
```json
[
  {"block_id": "p1_l0_c3f6aa9d", "content": "# Guideline"},
  {"block_id": "p1_l2_8f14e45f", "content": "## Subject: CAR"}
]
```

**LLM returns:**
```json
[
  {
    "block_id": "p1_l2_8f14e45f",
    "original": "CAR",
    "suggested": "Capital Adequacy Requirements",
    "reason": "Use full name",
    "confidence": "medium"
  }
]
```

**Perfect mapping!** ✅

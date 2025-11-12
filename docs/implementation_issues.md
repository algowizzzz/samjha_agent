# Implementation Issues Identified

## Issue 1: Single Response Mapping to Multiple Questions

**Problem**: When user provides a single response like "above 90%, PV01, latest date, limit to 10" that answers all 3 questions, the synthesis step only maps it to Q1, leaving Q2 and Q3 as "(not yet answered)".

**Evidence from logs**:
```
Synthesis: Extracted 1 user responses: ['above 90%, PV01, latest date, limit to 10']
Q1: What do you mean by 'large exposure'?
A1: above 90%, PV01, latest date, limit to 10
Q2: Would you like to see large exposures for a specific desk...
A2: (not yet answered)
Q3: Do you want to see the latest data...
A3: (not yet answered)
```

**Root Cause**: The synthesis step doesn't parse multi-part answers. It treats the entire response as answering only the first question.

**Impact**: Even when user provides complete information, synthesis fails to recognize it, leading to unnecessary clarification loops.

---

## Issue 2: Confirmation Phrase Interpretation

**Problem**: When user responds with "yes all of them" or "all, yes", the synthesis LLM isn't properly interpreting these as confirmations of all questions.

**Evidence from logs**:
- User provides "yes all of them"
- Synthesis extracts it but doesn't map it to all questions
- Clarification gate still asks for clarification

**Root Cause**: 
1. The synthesis prompt doesn't explicitly instruct how to handle confirmation phrases
2. The Q&A format shows "(not yet answered)" for Q2 and Q3, so the LLM doesn't know to apply defaults

**Impact**: Users get stuck in clarification loops even when they're trying to confirm all questions.

---

## Issue 3: Multiple Clarifications Not Properly Accumulated

**Problem**: When user provides multiple clarifications in sequence (e.g., first "above 90%...", then "yes all"), the system accumulates them but doesn't properly merge them.

**Evidence**:
- `accumulated_clarifications` list grows: `['above 90%, PV01, latest date, limit to 10', 'yes all of them']`
- But synthesis only uses the last one or doesn't properly combine them

**Root Cause**: The synthesis step receives all accumulated clarifications as separate entries but doesn't have logic to:
- Prioritize specific answers over confirmations
- Merge multiple clarifications intelligently
- Understand that later clarifications might override earlier ones

**Impact**: Confusion when user provides multiple clarifications - system doesn't know which to use.

---

## Issue 4: Synthesis Prompt Doesn't Parse Multi-Part Answers

**Problem**: The synthesis prompt doesn't instruct the LLM to parse answers like "above 90%, PV01, latest date, limit to 10" and map parts to different questions.

**Current Prompt**:
```
Combine specific answers from clarifications into the query
```

**Missing**: Instructions to:
- Parse comma-separated or multi-part answers
- Map different parts to different questions
- Recognize that "above 90%" answers Q1, "PV01" answers Q2, "latest date" answers Q3

**Impact**: Even when user provides complete information in one response, it's not properly utilized.

---

## Issue 5: Context Not Used in Synthesis

**Problem**: The synthesis step doesn't use the full context (conversation history, previous plans, etc.) to better understand clarifications.

**Current**: Synthesis only sees:
- Original query
- Clarification questions
- User responses

**Missing**: 
- Conversation history (might help understand context)
- Previous plans (might help understand what was tried before)
- Schema information (might help understand what "PV01" means)

**Impact**: Synthesis makes decisions without full context, leading to poor synthesis.

---

## Issue 6: Clarification Gate Evaluates Synthesized Query Too Strictly

**Problem**: Even after successful synthesis (e.g., "Find risk exposures with PV01 limits above 90% utilization..."), the clarification gate sometimes still asks for clarification.

**Evidence from logs**:
```
Synthesized query: Find risk exposures with PV01 limits above 90% utilization, limited to top 10 results, using the most recent available data
Clarification gate: needs_clarification=False  # This works!
```

But in other cases, even with good synthesis, gate still asks for clarification.

**Root Cause**: The clarification gate prompt might be too strict or not properly evaluating the synthesized query.

**Impact**: Unnecessary clarification requests even after good synthesis.

---

## Issue 7: Response Extraction Regex May Miss Edge Cases

**Problem**: The regex pattern for extracting user responses might not handle all formats correctly.

**Current Pattern**: `r'\n\nClarification\s+(\d+)\s*:\s*(.+?)(?=\n\nClarification\s+\d+|$)'`

**Potential Issues**:
- Doesn't handle if there's whitespace before the number
- Doesn't handle if response spans multiple lines
- Doesn't handle if there are special characters

**Impact**: Some user responses might not be extracted correctly.

---

## Issue 8: No Validation of Extracted Responses

**Problem**: After extracting user responses, there's no validation that:
- The number of responses matches the number of questions
- Responses are meaningful (not empty, not just whitespace)
- Responses actually answer the questions

**Impact**: System might proceed with incomplete or invalid responses.

---

## Issue 9: Synthesis Doesn't Handle Partial Answers

**Problem**: If user answers only some questions (e.g., only Q1 and Q3, skipping Q2), synthesis doesn't know how to handle the missing answers.

**Current Behavior**: Shows "(not yet answered)" for missing questions, but doesn't apply defaults.

**Impact**: System asks for clarification even when partial answers + defaults would be sufficient.

---

## Issue 10: Context Builder Uses `query["current"]` But It May Not Be Updated

**Problem**: In `build_node_context()`, we build `combined` query with clarifications, but this happens at context build time. If clarifications are added later, the context might be stale.

**Current Flow**:
1. `build_node_context()` builds `combined` from `accumulated_clarifications`
2. User provides new clarification
3. `replan_node` adds it to `accumulated_clarifications`
4. But context was already built, so new clarification might not be in `combined`

**Impact**: New clarifications might not be included in the context passed to tools.

---

## Summary of Critical Issues

### High Priority:
1. **Issue 1**: Single response not mapped to multiple questions
2. **Issue 2**: Confirmation phrases not interpreted correctly
3. **Issue 4**: Synthesis prompt doesn't parse multi-part answers

### Medium Priority:
4. **Issue 3**: Multiple clarifications not properly merged
5. **Issue 5**: Context not used in synthesis
6. **Issue 6**: Clarification gate too strict

### Low Priority:
7. **Issue 7**: Regex edge cases
8. **Issue 8**: No response validation
9. **Issue 9**: Partial answers handling
10. **Issue 10**: Stale context issue


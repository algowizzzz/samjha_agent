# Node Context Gap Analysis

## Current State: Information Available to Each Decision-Making Node

### 1. **planner_node** (Initial Planning)
**Receives:**
- ✅ `user_input` - Original query
- ✅ `table_schema` - Schema information
- ✅ `docs_meta` - Business glossary/docs
- ✅ `conversation_history` - Previous conversation turns
- ✅ `previous_clarifications: []` - Empty (first time)

**Missing:**
- ❌ None (appropriate for initial planning)

---

### 2. **replan_node** (Replanning After Clarification/Evaluation)
**Receives:**
- ✅ `user_input` - Original query
- ✅ `user_clarification` - User's clarification response
- ✅ `accumulated_clarifications` - All previous clarifications
- ✅ `clarification_questions` - Questions that were asked
- ✅ `table_schema` - Schema information
- ✅ `docs_meta` - Business glossary/docs
- ✅ `conversation_history` - Previous conversation turns
- ✅ `previous_clarifications` - Previous clarification questions

**Missing:**
- ❌ **`evaluator_notes`** - When replanning after evaluation, doesn't get feedback from evaluator
- ❌ **`previous_plan`** - Doesn't know what the previous plan was (could help avoid repeating mistakes)
- ❌ **`previous_execution_result`** - Doesn't know what results were returned (could help understand what went wrong)
- ❌ **`satisfaction`** - Doesn't know why evaluation failed (could help improve replan)

---

### 3. **evaluate_node** (Evaluating Query Results)
**Receives:**
- ✅ `user_input` - Original query
- ✅ `execution_result` - Query execution results
- ✅ `execution_stats` - Execution statistics
- ✅ `table_schema` - Schema information

**Missing:**
- ❌ **`conversation_history`** - Could help understand context of follow-up queries
- ❌ **`plan` / `plan_explain`** - Doesn't know what the planner intended (could help assess if plan matched intent)
- ❌ **`previous_evaluator_notes`** - If re-evaluating after replan, doesn't know what previous evaluation said
- ❌ **`replan_count`** - Doesn't know if this is first evaluation or after multiple replans (could adjust criteria)

---

### 4. **end_node** (Final Response Generation)
**Receives:**
- ✅ `user_input` - Original query
- ✅ `plan` - SQL plan
- ✅ `execution_result` - Query results
- ✅ `execution_stats` - Execution statistics
- ✅ `conversation_history` - Previous conversation turns
- ✅ `evaluator_notes` - Evaluation feedback
- ✅ `satisfaction` - Evaluation satisfaction level

**Missing:**
- ❌ None (has comprehensive context)

---

## Critical Gaps Identified

### **Gap 1: replan_node Missing Evaluation Feedback**
**Impact:** When replanning after evaluation fails, replan_node doesn't know:
- What the previous plan was
- What results were returned
- Why the evaluation failed
- What improvements were suggested

**Example Scenario:**
1. User: "top exposure"
2. Planner generates SQL
3. Execute returns results
4. Evaluator says "needs_work" with notes: "Results show only 3 rows, user likely wanted more context"
5. Replan_node replans but doesn't know the evaluator's feedback

**Fix Needed:** Pass `evaluator_notes`, `previous_plan`, `suggested_improvements` to replan_node

---

### **Gap 2: evaluate_node Missing Context**
**Impact:** Evaluator doesn't have:
- Conversation history (for follow-up queries like "average of these")
- Plan explanation (to understand planner's intent)
- Previous evaluation notes (if re-evaluating)

**Example Scenario:**
1. User: "top exposure" → Returns 3 results
2. User: "average of these" → Evaluator doesn't know "these" refers to the 3 results from previous query

**Fix Needed:** Pass `conversation_history`, `plan_explain` to evaluate_node

---

### **Gap 3: No Standardized Context Object**
**Impact:** Each node extracts context differently, leading to:
- Inconsistency in what information is available
- Potential for missing critical context
- Hard to maintain and extend

**Fix Needed:** Create a standardized context builder that all nodes use

---

## Recommendations

### **Priority 1: High Impact Fixes**

1. **Pass evaluation feedback to replan_node**
   - Add `evaluator_notes`, `suggested_improvements`, `previous_plan` to replan_node
   - Use this feedback to improve replanning

2. **Pass conversation_history to evaluate_node**
   - Help evaluator understand context for follow-up queries
   - Better assessment of whether results match user intent

3. **Pass plan_explain to evaluate_node**
   - Help evaluator understand what the planner intended
   - Better assessment of whether plan matched intent

### **Priority 2: Medium Impact Fixes**

4. **Track replan count in state**
   - Help evaluator adjust criteria if multiple replans occurred
   - Prevent infinite replan loops

5. **Standardize context passing**
   - Create a `build_node_context()` function
   - Ensures all nodes get consistent, complete context

### **Priority 3: Nice to Have**

6. **Add previous evaluation history**
   - Track all evaluations in a session
   - Help identify patterns

7. **Add plan evolution tracking**
   - Track how plans change across replans
   - Help identify common failure patterns


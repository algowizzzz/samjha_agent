# Phase 3: "Ask RiskGPT" Feature - Implementation Plan

## ✅ **Backend Complete!**

### **1. API Endpoint: `/api/doc_review/ask_riskgpt`**

**Location:** `external/routes/doc_review_routes.py`

**Request:**
```json
{
  "file_id": "doc_abc123",
  "selected_block_ids": ["p1_b3_abc", "p1_b4_def"],
  "user_prompt": "Make this more concise"
}
```

**Response:**
```json
{
  "file_id": "doc_abc123",
  "suggestions": [
    {
      "block_id": "p1_b3_abc",
      "original": "The purpose of this draft policy is to outline...",
      "suggested": "This policy outlines...",
      "reason": "Made more concise as requested",
      "confidence": "high"
    }
  ],
  "selected_block_ids": ["p1_b3_abc", "p1_b4_def"],
  "user_prompt": "Make this more concise"
}
```

---

### **2. LLM Function: `ask_riskgpt_for_blocks()`**

**Location:** `external/doc_review/llm.py`

**Features:**
- Takes selected blocks + user prompt + full document context
- Sends to Claude with structured prompt
- Returns validated suggestions with block IDs
- Handles JSON parsing and error cases

**LLM Prompt:**
```
You are RiskGPT, an expert document improvement assistant.

CRITICAL RULES:
1. Update ONLY the selected blocks provided
2. Use the full document as context for consistency
3. Preserve markdown formatting
4. Return structured JSON with block_id and new_content
5. Provide a clear reason for each change
6. Be conservative - only change what the user asks for
```

---

### **3. Frontend API Call**

**Location:** `Doc Review Workspace Wireframe/src/lib/api.ts`

**New Types:**
```typescript
export type RiskGPTSuggestion = {
  block_id: string;
  original: string;
  suggested: string;
  reason: string;
  confidence: 'high' | 'medium' | 'low';
};
```

**New Function:**
```typescript
export async function askRiskGPT(
  fileId: string,
  selectedBlockIds: string[],
  userPrompt: string
): Promise<{ 
  file_id: string; 
  suggestions: RiskGPTSuggestion[]; 
  selected_block_ids: string[]; 
  user_prompt: string 
}>;
```

---

## 🚧 **Frontend UI - TODO**

### **What Needs to Be Built:**

#### **1. Block Selection UI**

**Requirements:**
- Click a block → Select it (highlight with border)
- Shift + Click → Multi-select
- Cmd/Ctrl + Click → Add to selection
- Click elsewhere → Deselect all

**Visual:**
```
┌────────────────────────────────────────┐
│ The purpose of this policy...          │  ← Normal block
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
▌ The purpose of this policy...          │  ← Selected (blue outline)
└────────────────────────────────────────┘
  [Ask RiskGPT]
```

---

#### **2. Inline Chat Input**

**Requirements:**
- Appears when 1+ blocks selected
- Small text input below selected blocks
- "Ask RiskGPT" button
- Placeholder: "Ask RiskGPT to improve..."

**Visual:**
```
┌────────────────────────────────────────┐
▌ The purpose of this policy...          │  ← Selected block
└────────────────────────────────────────┘
┌────────────────────────────────────────┐
│ [Ask RiskGPT to improve...]            │  ← Input
│                         [Ask RiskGPT ✨]│  ← Button
└────────────────────────────────────────┘
```

---

#### **3. AI Suggestions Display**

**Requirements:**
- Blue left border on blocks with AI suggestions
- Hover tooltip showing suggestion details
- Accept/Reject buttons

**Visual:**
```
┌────────────────────────────────────────┐
▌ This policy outlines...                │  ← Blue left border (AI suggestion)
└────────────────────────────────────────┘
  💡 RiskGPT: "Made more concise as requested"
  ✓ Accept  ✗ Reject
```

---

#### **4. AI Applied State**

**Requirements:**
- Purple left border after accepting
- Change history updated

**Visual:**
```
┌────────────────────────────────────────┐
▌ This policy outlines...                │  ← Purple left border (AI applied)
└────────────────────────────────────────┘
```

---

## 📋 **Implementation Steps (Frontend)**

### **Step 1: Add Block Selection State**

```typescript
const [selectedBlockIds, setSelectedBlockIds] = useState<Set<string>>(new Set());

const handleBlockClick = (blockId: string, event: React.MouseEvent) => {
  if (event.shiftKey || event.metaKey || event.ctrlKey) {
    // Multi-select
    setSelectedBlockIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(blockId)) {
        newSet.delete(blockId);
      } else {
        newSet.add(blockId);
      }
      return newSet;
    });
  } else {
    // Single select
    setSelectedBlockIds(new Set([blockId]));
  }
};
```

---

### **Step 2: Add Inline Chat Input**

```typescript
const [riskGPTPrompt, setRiskGPTPrompt] = useState('');
const [isAskingRiskGPT, setIsAskingRiskGPT] = useState(false);

const handleAskRiskGPT = async () => {
  if (!riskGPTPrompt.trim() || selectedBlockIds.size === 0) return;
  
  setIsAskingRiskGPT(true);
  try {
    const result = await askRiskGPT(
      fileId,
      Array.from(selectedBlockIds),
      riskGPTPrompt
    );
    
    // Apply suggestions to blocks
    applyRiskGPTSuggestions(result.suggestions);
    
    setRiskGPTPrompt('');
    setSelectedBlockIds(new Set());
  } catch (error) {
    console.error('RiskGPT failed:', error);
  } finally {
    setIsAskingRiskGPT(false);
  }
};
```

---

### **Step 3: Update Block Interface for AI Suggestions**

```typescript
interface Block {
  id: string;
  type: BlockType;
  content: string;
  changeType: ChangeType;
  commentCount: number;
  suggestion?: VerificationSuggestion;
  aiSuggestion?: RiskGPTSuggestion;  // NEW
  changeHistory: ChangeRecord[];
}
```

---

### **Step 4: Add Accept/Reject for AI Suggestions**

```typescript
const acceptAISuggestion = (blockId: string) => {
  setBlocks(prev => prev.map(b => {
    if (b.id === blockId && b.aiSuggestion) {
      const newChangeRecord: ChangeRecord = {
        timestamp: new Date().toISOString(),
        type: 'ai_applied',
        original: b.content,
        modified: b.aiSuggestion.suggested,
        reason: `Accepted RiskGPT: ${b.aiSuggestion.reason}`,
        user: 'user'
      };
      return {
        ...b,
        content: b.aiSuggestion.suggested,
        changeType: 'ai_applied',
        aiSuggestion: undefined,
        changeHistory: [...b.changeHistory, newChangeRecord]
      };
    }
    return b;
  }));
};

const rejectAISuggestion = (blockId: string) => {
  setBlocks(prev => prev.map(b => {
    if (b.id === blockId && b.aiSuggestion) {
      const newChangeRecord: ChangeRecord = {
        timestamp: new Date().toISOString(),
        type: 'rejected',
        original: b.content,
        modified: b.content,
        reason: `Rejected RiskGPT: ${b.aiSuggestion.reason}`,
        user: 'user'
      };
      return {
        ...b,
        changeType: 'rejected',
        aiSuggestion: undefined,
        changeHistory: [...b.changeHistory, newChangeRecord]
      };
    }
    return b;
  }));
};
```

---

## 🎨 **Visual Design**

### **Selection State:**
- Selected blocks: `border-2 border-blue-400 bg-blue-50`
- Unselected: Normal styling

### **AI Suggestion State:**
- Blue left border: `border-l-4 border-blue-500 bg-blue-50`
- Hover tooltip with suggestion details

### **AI Applied State:**
- Purple left border: `border-l-4 border-purple-500 bg-purple-50`

---

## 📊 **Complete Workflow**

```
1. User selects 2 blocks (shift + click)
   ↓
2. Inline chat input appears
   ↓
3. User types: "Make this more concise"
   ↓
4. Click "Ask RiskGPT ✨"
   ↓
5. Loading spinner appears
   ↓
6. Backend calls Claude with blocks + prompt + full doc
   ↓
7. Claude returns suggestions
   ↓
8. Blocks turn BLUE (ai_suggested)
   ↓
9. User hovers → sees suggestion details
   ↓
10. User clicks "Accept" → Block turns PURPLE (ai_applied)
    OR
    User clicks "Reject" → Block turns RED (rejected)
```

---

## 🔄 **Multiple Change Indicators (Phase 2.5)**

**Current:** Each block shows ONE color (latest change)
**Proposed:** Show ALL active changes as badges

**Example:**
```
┌────────────────────────────────────────┐
▌ The purpose of this policy...  🟡 🔵  │  ← Yellow + Blue badges
└────────────────────────────────────────┘
```

**Implementation:**
- Add badges array to Block interface
- Render small colored circles/icons for each active change
- Click badge → show details in tooltip

---

## ✅ **What's Done:**
- ✅ Backend API endpoint
- ✅ LLM function with structured prompt
- ✅ Frontend API call types

## 🚧 **What's Next:**
- [ ] Block selection UI (shift/cmd + click)
- [ ] Inline chat input component
- [ ] AI suggestion display with accept/reject
- [ ] Multiple change indicators (badges)
- [ ] User edit tracking (green border)
- [ ] End-to-end testing

---

**Ready to implement the frontend UI!** 🚀


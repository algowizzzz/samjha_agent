
# AI BULK DOC ANALYSIS  
## UI DESIGN MASTER SPEC — HIGH-FIDELITY WIREFRAMES (FINAL)

**Audience:** UI / UX Designers (Figma)  
**Platform:** Desktop Web App (No Mobile)  
**Design Goal:** Deterministic, professional, audit-grade UI  
**Scope:** Panels 1–3, all states, all components, pixel-accurate

---

## 1. DESIGN PHILOSOPHY (NON-NEGOTIABLE)

1. Determinism over delight  
2. One primary action per panel  
3. Progress visibility at all times  
4. Failure must be visible and isolated  
5. No conversational UI metaphors  

---

## 2. GLOBAL FRAME & GRID SYSTEM

### Viewport
- Width: 1440–1600px
- Height: 100vh
- Per-panel scrolling only
- Desktop only

### Three-Panel Layout
- Panel 1 (Documents): 22%
- Panel 2 (Prompt Chain): 34%
- Panel 3 (Run & Output): 44%
- Fixed, non-collapsible

### Global Header
- Height: 56px
- App name + session ID (read-only)
- User avatar with menu
- Divider: 1px neutral-200

---

## 3. DESIGN TOKENS

### Typography
- H1: 20px / 600
- H2: 16px / 600
- Body: 14px / 400
- Meta: 12px / 400
- Mono: 13px / 400
- Font: Inter

### Spacing (8pt system)
4 / 8 / 16 / 24 / 32

### Colors
- Background: Neutral-50
- Border: Neutral-200
- Primary: Blue-600
- Success: Green-600
- Error: Red-600
- Disabled: Neutral-300

---

## 4. PANEL 1 — DOCUMENTS

- Input type selector (PDF enabled, others disabled)
- Upload PDFs (auto-start)
- Document table with status badges
- Error-only delete action
- Slide-in document detail drawer (420px)
- Empty state: Upload PDF documents to begin

---

## 5. PANEL 2 — PROMPT CHAIN

- Saved chain library (cards)
- Card states: default, hover, selected, invalid
- Read-only chain detail view
- Ordered step list (always visible)
- Empty state: Select a saved chain

---

## 6. PANEL 3 — RUN & OUTPUT

- Run Chain primary CTA
- Disabled until docs converted + chain selected
- Run progress table (per document)
- Token visibility (input / output)
- Sticky completion summary
- Download all / per document
- Empty state: Ready to run

---

## 7. STATES & FEEDBACK

- Skeleton loaders only
- Inline errors
- Toasts for system errors
- Partial success allowed

---

## 8. REUSABLE COMPONENTS

- Header
- Panels
- Segmented control
- Buttons
- Status badges
- Tables
- Chain cards
- Drawer
- Toasts
- Empty states
- Confirmation modal

---

## 9. DESIGNER ACCEPTANCE CRITERIA

Design is approved only if:
1. No explanation required
2. Disabled states have reasons
3. Tokens are visible
4. Errors are isolated
5. No chat metaphors
6. One primary action per panel
7. All backend states are visible

---

## 10. FIGMA DELIVERABLES

- Full desktop layout
- All panel states
- All modals and drawers
- Component library
- Token reference page

---

**This document is the single source of truth for UI design.**

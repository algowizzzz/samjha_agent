
# AI BULK DOC ANALYSIS  
## FRONTEND ACCEPTANCE TESTS, FIGMA AUTO-LAYOUT CHECKLIST & DESIGN-SYSTEM JSON

**Audience:** Frontend Engineers & UI / UX Designers  
**Purpose:** Eliminate ambiguity between design and implementation  
**Status:** Authoritative companion to UI Design Master Spec

---

# SECTION 1 — FRONTEND ACCEPTANCE TESTS (MAPPED TO UI STATES)

This section defines **testable acceptance criteria** for frontend engineers.  
Each test must be verifiable via UI behavior alone (no backend assumptions).

---

## 1. GLOBAL APPLICATION

### G-01 App Loads
**Given** the application loads  
**Then**
- Three panels are visible (Documents, Prompt Chain, Run & Output)
- Global header is visible with app name and session ID
- No panel is collapsed or reordered

---

## 2. PANEL 1 — DOCUMENTS

### P1-01 Empty State
**Given** no documents uploaded  
**Then**
- Empty state text is visible: “Upload PDF documents to begin”
- Upload button is enabled
- Document table is not rendered

---

### P1-02 Upload PDFs
**Given** user clicks “Upload PDFs”  
**When** multiple PDFs are selected  
**Then**
- Files appear immediately in document table
- Status shows QUEUED → PROCESSING
- Conversion starts automatically (no extra click)

---

### P1-03 Conversion Success
**Given** conversion completes  
**Then**
- Status badge changes to CONVERTED
- Document row remains clickable
- No delete icon is shown

---

### P1-04 Conversion Error
**Given** conversion fails  
**Then**
- Status badge shows ERROR
- Trash icon appears
- Error message visible in document drawer

---

### P1-05 Delete Errored Document
**Given** user clicks delete on ERROR row  
**Then**
- Confirmation modal appears
- On confirm, document is removed from list
- No other documents are affected

---

## 3. PANEL 2 — PROMPT CHAIN

### P2-01 Empty State
**Given** no chain selected  
**Then**
- Empty state text: “Select a saved chain to continue”
- Run button remains disabled

---

### P2-02 Chain Selection
**Given** user selects a valid chain  
**Then**
- Chain card shows selected state
- Chain detail view renders step list
- Step order is visible and fixed

---

### P2-03 Invalid Chain
**Given** a chain is incomplete  
**Then**
- Chain card appears disabled
- Warning icon and label are visible
- Chain cannot be selected

---

## 4. PANEL 3 — RUN & OUTPUT

### P3-01 Run Disabled
**Given**
- At least one document is not CONVERTED
- OR no valid chain selected  
**Then**
- Run Chain button is disabled
- Tooltip explains why

---

### P3-02 Run Execution
**Given** all prerequisites met  
**When** user clicks Run Chain  
**Then**
- Run progress table appears
- Each document shows step progress (R0…Rn)
- Status updates live

---

### P3-03 Token Visibility
**Given** a step completes  
**Then**
- Tokens column shows input/output values
- Values are readable and formatted as “in / out”

---

### P3-04 Partial Failure
**Given** some documents fail  
**Then**
- Failed rows show ERROR
- Successful rows show SUCCESS
- Download buttons remain enabled for successful rows

---

### P3-05 Download Outputs
**Given** document status is SUCCESS  
**Then**
- Per-document download button is enabled
- Download all outputs button downloads only successful files

---

# SECTION 2 — FIGMA AUTO-LAYOUT CHECKLIST

This checklist **must be followed exactly** when building Figma frames.

---

## A. Global Layout

- [ ] Root frame width set to 1440px or 1600px
- [ ] Three-column layout using Auto Layout (horizontal)
- [ ] Panels use fixed width percentages (22 / 34 / 44)
- [ ] Each panel scrolls independently
- [ ] Global header is fixed (not scrolling)

---

## B. Panels

- [ ] Each panel is its own Auto Layout frame (vertical)
- [ ] Panel padding uses spacing tokens only (8pt system)
- [ ] Panel headers are fixed within panel
- [ ] Content area scrolls independently

---

## C. Tables & Lists

- [ ] Tables use Auto Layout rows
- [ ] Column widths are fixed, not hug-content
- [ ] Row hover states defined
- [ ] Status badges are components, not text

---

## D. Buttons & Controls

- [ ] Primary and secondary buttons are components
- [ ] Disabled states are variants, not overrides
- [ ] Tooltips are separate components
- [ ] Icons are vector-based, not raster

---

## E. Drawers & Modals

- [ ] Drawer uses overlay frame aligned right
- [ ] Drawer width fixed at 420px
- [ ] Background dim applied on open
- [ ] Confirmation modal is centered and reusable

---

## F. States

- [ ] Empty states are their own components
- [ ] Loading skeletons exist for each panel
- [ ] Error states are visually distinct but consistent

---

# SECTION 3 — DESIGN-SYSTEM JSON (FOR FRONTEND)

This JSON is a **reference contract**.  
Frontend may transform it, but values must match.

```json
{
  "typography": {
    "fontFamily": "Inter, system-ui",
    "h1": { "size": 20, "weight": 600 },
    "h2": { "size": 16, "weight": 600 },
    "body": { "size": 14, "weight": 400 },
    "meta": { "size": 12, "weight": 400 },
    "mono": { "size": 13, "weight": 400 }
  },
  "spacing": {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32
  },
  "colors": {
    "background": "#F9FAFB",
    "panel": "#FFFFFF",
    "border": "#E5E7EB",
    "primary": "#2563EB",
    "success": "#16A34A",
    "error": "#DC2626",
    "disabled": "#D1D5DB"
  },
  "radius": {
    "sm": 4,
    "md": 6,
    "lg": 8
  },
  "components": {
    "button": {
      "height": 40,
      "radius": 6,
      "primary": "colors.primary",
      "disabled": "colors.disabled"
    },
    "drawer": {
      "width": 420
    },
    "header": {
      "height": 56
    }
  }
}
```

---

## FINAL NOTE

If:
- UI design matches the Design Master Spec
- Acceptance tests pass visually
- Auto-layout checklist is fully satisfied

Then frontend build risk is **minimal and predictable**.

This document, together with the UI Design Master Spec, completes UI → FE handoff.

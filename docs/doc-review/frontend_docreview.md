# **Document Review Frontend – Comprehensive UI/UX & Implementation Specification**

**Version:** 2.0
**Date:** November 16, 2025
**Purpose:** A complete specification for building, testing, and deploying a production-grade web frontend for the Document Review Agent, ensuring tight integration with the existing backend APIs and WebSocket event streams. 

> Wireframe Source of Truth
>
> - Use the “Doc Review Workspace Wireframe” folder as the visual source of truth (layouts, spacing, typography, component states). No deviations or surprises — implement the UI as-is from this wireframe.
> - If any conflict arises between this spec and the wireframe, defer to the wireframe for visuals and to this spec for routes/data/behaviors. Raise discrepancies explicitly before changing either.

---

## **Table of Contents**

1.  [Executive Summary](#1-executive-summary)
2.  [Frontend Architecture](#2-frontend-architecture)
3.  [UI/UX Design Principles](#3-uiux-design-principles)
4.  [Information Architecture & Screens](#4-information-architecture--screens)
5.  [Screen Specifications](#5-screen-specifications)
    -   5.1 Documents List (`/documents`)
    -   5.2 Review Workspace (`/documents/:file_id`)
6.  [Component & State Management](#6-component--state-management)
7.  [API & WebSocket Integration](#7-api--websocket-integration)
8.  [Implementation Plan](#8-implementation-plan)
9.  [Non-Functional Requirements](#9-non-functional-requirements)
10. [Detailed Implementation Guide](#10-detailed-implementation-guide)
    -   10.1 Top-Level File Structure
    -   10.2 API Layer (`src/api`)
    -   10.3 App Setup (`src/app`)
    -   10.4 Layout Components (`src/components/layout`)
    -   10.5 Documents List Components (`src/components/documents`)
    -   10.6 Workspace Components (`src/components/workspace`)
    -   10.7 Custom Hooks (`src/hooks`)
    -   10.8 Type Definitions (`src/types`)


---

## **1. Executive Summary**

### **1.1 Project Goal**

This document outlines the specification for a modern, focused web user interface for the Document Review Agent. The frontend will provide an intuitive, enterprise-friendly experience for non-technical users, such as risk and policy owners, to manage, review, and improve policy documents. It will serve as the primary interface for interacting with the powerful backend agent, making the entire workflow accessible and transparent.

### **1.2 Key User Outcomes**

-   **Seamless Document Management:** Easily upload new documents or register existing ones from a server path.
-   **Transparent Workflow Execution:** Run review workflows (Phase 1, 2, 4) with clear, real-time progress indicators and status updates.
-   **Actionable Insights:** Review structured issues, improvement suggestions, and analysis reports in a clear, organized manner.
-   **Interactive Editing:** Directly edit documents in a rich, contextual editor with integrated AI assistance ("Ask AI about this selection").
-   **High-Level Guidance:** Use a dedicated chat panel for higher-level instructions, questions, and agent guidance.

### **1.3 Core Frontend Mandates**

-   **API-Driven:** The frontend is a pure client for the existing `/api/doc_review/*` REST endpoints.
-   **Real-time:** Leverages Socket.IO events for live status updates, activity logs, and VFS file changes.
-   **Structured Layout:** Adopts a three-pane "review workspace" (Navigation/Issues, Editor/Diff, Chat/Activity) for optimal cognitive load.
-   **Professional & Clean:** The UI will be minimal, responsive, and aligned with modern enterprise application design standards.

---

## **2. Frontend Architecture**

### **2.1 Technology Stack**

-   **Framework:** React (using Vite or Next.js for the build system).
-   **Routing:** React Router for managing navigation between screens.
-   **Styling:** Tailwind CSS complemented by a headless component library like `shadcn/ui` for consistency and rapid development.
-   **State Management:**
    -   **Server State:** TanStack Query (React Query) for caching, refetching, and managing all data from the backend APIs.
    -   **Local UI State:** Minimal local state (`useState`, `useReducer`) for managing component-level concerns like tab selection or modal visibility.
-   **Rich Text Editor:** A robust, React-based Markdown editor such as Tiptap, Novel, or a block-based editor to provide a seamless editing experience.
-   **Real-time Communication:** The official Socket.IO client library to connect to the backend and handle real-time events.

### **2.2 High-Level Component Structure**

```text
src/
├── api/              # API client layer for all backend calls
├── app/              # Routing, query client setup
├── components/
│   ├── layout/       # Main app shell (Sidebar, TopBar)
│   ├── common/       # Reusable components (Buttons, Modals, Tables)
│   ├── documents/    # Components for the Documents List screen
│   └── workspace/    # Components for the three-pane Review Workspace
├── hooks/            # Custom hooks for sockets, editor state, etc.
└── types/            # TypeScript types for API responses and state
```

This structure separates concerns cleanly, making the application easier to develop, test, and maintain.

---

## **3. UI/UX Design Principles**

1.  **Clarity Over Cleverness:** The user must always understand the document's current state, what has been done, what is happening now, and what the next logical steps are.
2.  **Review-First, AI-Second:** The AI is an assistant. It provides summaries, suggestions, and automates tedious tasks. The user is always in control of the final output.
3.  **Contextual Actions:** Interactions should be intuitive. Selecting text should immediately present relevant actions like "Ask AI about this" or "Improve wording."
4.  **Minimal Cognitive Load:** The main workspace is constrained to a three-pane layout to prevent overwhelming the user. Modals and deep navigation are used sparingly.
5.  **Traceability:** All changes and system actions are logged and visible. The user can easily see how the document has evolved using diff views, issue lists, and the activity log.

---

## **4. Information Architecture & Screens**

### **4.1 Main Screens**

1.  **Documents List (`/documents`):** The application's entry point. Provides a high-level overview of all documents and their review status.
2.  **Review Workspace (`/documents/:file_id`):** The core interactive screen where a single document is analyzed, reviewed, and edited.
3.  **Templates (`/templates`):** (v1 - View Only) A simple screen to browse and preview available review templates.

### **4.2 Review Workspace Layout**

The workspace is divided into three persistent panes to provide a comprehensive review environment:

-   **Left Pane:** Navigation and Analysis (Outline, Issues, Artifacts).
-   **Center Pane:** Document Content (Editor, Diff Viewer).
-   **Right Pane:** Agent Interaction (Chat, Live Activity Log).

---

## **5. Screen Specifications**

### **5.1 Documents List (`/documents`)**

**Purpose:** To allow users to view, manage, and initiate reviews for their documents.

-   **Layout:**
    -   **Header:** Title ("Documents"), primary action ("Upload & Review"), secondary action ("Register by Path").
    -   **Controls:** Search input and status filters (`All`, `In Progress`, `Completed`, `Failed`).
    -   **Table View:** A list of documents with columns for Name, Source, Status, Issue Counts, and Last Updated.
-   **Interactions:**
    -   Clicking a row navigates to the `Review Workspace` for that document.
    -   The "Upload & Review" flow opens a modal to upload a file, register it, and then navigate to the new workspace.
-   **API Integration:**
    -   `GET /api/doc_review/documents` to populate the table.
    -   `POST /api/doc_review/upload` followed by `POST /api/doc_review/documents` for the upload flow.

### **5.2 Review Workspace (`/documents/:file_id`)**

**Purpose:** To provide a focused, interactive environment for the detailed review and editing of a single document.

#### **5.2.1 Page Header**

-   **Content:** Document title, status badges for each phase (`Idle`, `Running`, `Done`, `Failed`).
-   **Actions:** Buttons to trigger backend workflows: "Run Full Review", "Run Phase 1", "Run Phase 2", "Assemble Final (Phase 4)".

#### **5.2.2 Left Pane (Navigation & Analysis)**

-   **Tab A: Outline:** A tree view of the document's `H1/H2/H3` heading structure, generated from the editor's content. Clicking an item scrolls the editor to that section.
-   **Tab B: Issues:** A filterable table of structured findings from the backend analysis (e.g., from `phase1_reports` or VFS JSON files). Columns include Section, Severity, and Description. Clicking an issue scrolls the editor to the relevant context.
-   **Tab C: Artifacts:** A file explorer view of the Virtual File System (VFS) for the current document. Allows users to browse and view all generated files (summaries, reports, section extractions). Powered by `GET /api/doc_review/vfs/tree` and `GET /api/doc_review/vfs/file`.

#### **5.2.3 Center Pane (Document Content)**

-   **Mode 1: Editor:** A rich text Markdown editor displaying the primary document artifact (e.g., `/phase4/final.md`).
    -   **Loading:** Content is loaded from the VFS via `GET /api/doc_review/vfs/file`.
    -   **Saving:** Changes are auto-saved on a debounce to the backend using `PATCH /api/doc_review/vfs/file`.
    -   **Real-time Updates:** Listens for `doc_review:vfs_file_updated` events and prompts the user to reload if the file is changed by the backend.
-   **Mode 2: Diff Viewer:** A toggleable side-by-side view comparing two versions of the document, typically the original vs. the latest edited version (e.g., `/original/document.md` vs. `/phase4/final.md`).

#### **5.2.4 Right Pane (Agent Interaction)**

-   **Tab 1: Chat:** An interactive chat interface for communicating with the Document Review Agent.
    -   **Initial State:** Populated with a welcome message from `GET /api/doc_review/welcome`.
    -   **Sending Messages:** `POST /api/doc_review/chat/<file_id>` with the user's message and any selected text from the editor.
    -   **Suggestions:** May include buttons to "Apply Suggestion," which would directly modify the document in the editor.
-   **Tab 2: Activity / Logs:** A real-time feed of events from the backend, powered by Socket.IO.
    -   Displays high-level `doc_review:status` events (e.g., "Phase 1 started...").
    -   Shows verbose `doc_review:log` messages in a collapsible section.
    -   Announces file updates from `doc_review:vfs_file_updated` events.

---

## **6. Component & State Management**

The frontend will be built with a clear component hierarchy. Page-level components (`WorkspacePage.tsx`) will be responsible for fetching data and managing high-level state, while presentation components (`IssuesPanel.tsx`) will receive data via props and handle user interactions.

Custom hooks like `useDocReviewSocket.ts` will encapsulate complex logic, such as managing the WebSocket connection and event listeners for a given `file_id`, keeping the page components clean and focused on layout and data flow.

---

## **7. API & WebSocket Integration**

### **7.0 Wireframe → API Mapping (Front ↔ Back)**

This maps each area of the “Doc Review Workspace Wireframe” to the backend API it uses.

- Documents List (screen `/documents`)
  - Data table: `GET /api/doc_review/documents`
  - Upload flow: `POST /api/doc_review/upload` → `POST /api/doc_review/documents`
  - Register by server path: `POST /api/doc_review/documents` (with `source_path`)
  - Upload dir browser (optional): `GET /api/doc_review/upload_dir/files`
  - Templates list (optional info): `GET /api/doc_review/templates`

- Workspace Header (screen `/documents/:file_id`)
  - Document state: `GET /api/doc_review/documents/:file_id`
  - Phase triggers: `POST /api/doc_review/documents/:file_id/run`, `/run_phase1`, `/run_phase2`, `/run_phase4`
  - Config updates: `PATCH /api/doc_review/documents/:file_id/config`

- Left Pane Tabs
  - Outline: derived client-side from editor content (no direct API)
  - Issues (Phase 1 outputs): 
    - Summary: `GET /api/doc_review/documents/:file_id/phase1_summary`
    - Reports: `GET /api/doc_review/documents/:file_id/phase1_reports`
  - Artifacts (VFS):
    - List: `GET /api/doc_review/vfs/tree?file_id=...&path=/`
    - File stat (optional): `GET /api/doc_review/vfs/stat?file_id=...&path=...`
    - Read: `GET /api/doc_review/vfs/file?file_id=...&path=...`

- Center Pane
  - Editor (Markdown primary artifact, e.g., `/phase4/final.md`):
    - Load content: `GET /api/doc_review/vfs/file`
    - Save changes: `PATCH /api/doc_review/vfs/file`
  - Diff Viewer (original vs final):
    - Read both files from VFS: `GET /api/doc_review/vfs/file` (twice, with paths)

- Right Pane Tabs
  - Chat:
    - Welcome: `GET /api/doc_review/welcome`
    - Send message: `POST /api/doc_review/chat/:file_id`
  - Activity / Logs (real-time):
    - Token: `GET /api/doc_review/token`
    - Socket events: `doc_review:join`, `doc_review:leave`, `doc_review:status`, `doc_review:log`, `doc_review:vfs_file_updated`

- Templates (screen `/templates`, view-only)
  - List: `GET /api/doc_review/templates`
  - Detail: `GET /api/doc_review/templates/:template_id`
  - Fitness scoring (optional): `POST /api/doc_review/documents/:file_id/template_fitness`

### **7.1 API Endpoint Mapping**

| UI Area | API Endpoints |
| :--- | :--- |
| Documents List | `GET /documents` |
| Upload & Register | `POST /upload` → `POST /documents` |
| Document Config | `GET /documents/<file_id>`, `PATCH /documents/<file_id>/config` |
| Run Workflows | `POST /documents/<file_id>/run`, `/run_phase1`, `/run_phase2`, `/run_phase4` |
| Phase 1 Reports | `GET /documents/<file_id>/phase1_summary`, `/phase1_reports` |
| VFS Operations | `GET /vfs/tree`, `GET /vfs/stat`, `GET /vfs/file`, `PATCH /vfs/file` |
| Chat | `GET /welcome`, `POST /chat/<file_id>` |
| Real-time Token | `GET /token` |

### **7.2 Real-time Event Handling**

-   **Connection:** On entering the workspace, the client will fetch a token via `GET /token`, connect to the Socket.IO server, and join the appropriate room by emitting `doc_review:join` with the `{ file_id }`.
-   **Event Listeners:** The client will listen for:
    -   `doc_review:status`: To update phase badges and the activity log with high-level progress.
    -   `doc_review:log`: To display detailed logs.
    -   `doc_review:vfs_file_updated`: To notify the user of changes to the open document or refresh the artifacts list.
-   **Disconnection:** On leaving the workspace, the client will emit `doc_review:leave` to clean up server-side resources.

---

## **8. Implementation Plan**

1.  **Phase 0 - Foundation:** Set up the React application, routing, API client layer, and basic layout shell. Establish the Socket.IO connection manager.
2.  **Phase 1 - Document Management:** Build the `Documents List` screen, including the data table and the complete "Upload & Register" user flow.
3.  **Phase 2 - Core Workspace:** Implement the three-pane `Review Workspace` layout. Wire up the center editor to the VFS, the header buttons to the run-phase APIs, and the Artifacts tab.
4.  **Phase 3 - Interactive Panels:** Build out the Chat and Activity panels in the right pane, and the Outline and Issues tabs in the left pane.
5.  **Phase 4 - Polish & Refinement:** Add the Diff viewer, implement loading states, error handling, user notifications, and conduct thorough UX testing.

---

## **9. Non-Functional Requirements**

-   **Performance:** The editor must remain responsive with documents up to 20 pages long. API requests should be efficiently cached.
-   **Resilience:** The application must handle API errors and WebSocket disconnections gracefully, with clear user feedback and automatic reconnection attempts.
-   **Security:** All API calls must be authenticated. No sensitive document content should be logged to the browser console.
-   **Extensibility:** The UI should be designed to accommodate new phases, reports, or artifacts from the backend with minimal redesign.

---

## **10. Detailed Implementation Guide**

This section provides a concrete, implementation-ready frontend structure that can be handed to a developer. It assumes the following stack:

*   **React + Vite** (or Create React App)
*   **React Router** for routing
*   **TanStack Query (React Query)** for data fetching
*   **Socket.IO client** for realtime

### **10.1 Top-Level File Structure**

```text
src/
  api/
    client.ts
    docReview.ts
  app/
    router.tsx
    queryClient.ts
  components/
    layout/
      AppLayout.tsx
      Sidebar.tsx
      TopBar.tsx
    documents/
      DocumentsPage.tsx
      DocumentsTable.tsx
      DocumentRowStatus.tsx
      UploadDialog.tsx
    workspace/
      WorkspacePage.tsx
      WorkspaceHeader.tsx
      LeftPane/
        LeftPaneTabs.tsx
        OutlinePanel.tsx
        IssuesPanel.tsx
        ArtifactsPanel.tsx
      CenterPane/
        CenterPane.tsx
        EditorPane.tsx
        DiffToggle.tsx
        DiffView.tsx
      RightPane/
        RightPaneTabs.tsx
        ChatPanel.tsx
        ActivityPanel.tsx
    templates/
      TemplatesPage.tsx
      TemplateList.tsx
      TemplatePreview.tsx
    common/
      Button.tsx
      Badge.tsx
      Tabs.tsx
      Modal.tsx
      Table.tsx
      Spinner.tsx
      ToastProvider.tsx
  hooks/
    useDocReviewSocket.ts
    useEditorSelection.ts
    useDocStatus.ts
  types/
    docReview.ts
  index.tsx
  main.css
```

### **10.2 API Layer (`src/api`)**

#### **10.2.1 `client.ts`**

A small wrapper around `fetch` or Axios with base URL, authentication headers, and standardized error handling.

```ts
// src/api/client.ts
export async function apiGet<T>(url: string): Promise<T> { /* ... */ }
export async function apiPost<T>(url: string, body?: any): Promise<T> { /* ... */ }
export async function apiPatch<T>(url: string, body?: any): Promise<T> { /* ... */ }
export async function apiPut<T>(url: string, body?: any): Promise<T> { /* ... */ }
```

#### **10.2.2 `docReview.ts`**

Contains typed functions for every doc review backend endpoint.

```ts
// src/api/docReview.ts
import { apiGet, apiPost, apiPatch, apiPut } from "./client";
import {
  DocumentRecord,
  Phase1Summary,
  VfsTreeItem,
  VfsFile,
  ChatMessageRequest,
  ChatMessageResponse,
} from "../types/docReview";

export function listDocuments() {
  return apiGet<DocumentRecord[]>("/api/doc_review/documents");
}

export function getDocument(fileId: string) {
  return apiGet<DocumentRecord>(`/api/doc_review/documents/${fileId}`);
}

export function uploadFile(formData: FormData) {
  return apiPost<{ saved_path: string }>("/api/doc_review/upload", formData);
}

export function registerDocument(payload: { source_path: string; name?: string }) {
  return apiPost<DocumentRecord>("/api/doc_review/documents", payload);
}

export function runFullWorkflow(fileId: string) {
  return apiPost(`/api/doc_review/documents/${fileId}/run`);
}

// ... other run_phaseX functions

export function getVfsTree(fileId: string, path = "/") {
  return apiGet<VfsTreeItem[]>(`/api/doc_review/vfs/tree?file_id=${fileId}&path=${encodeURIComponent(path)}`);
}

export function getVfsFile(fileId: string, path: string) {
  return apiGet<VfsFile>(`/api/doc_review/vfs/file?file_id=${fileId}&path=${encodeURIComponent(path)}`);
}

export function patchVfsFile(payload: { file_id: string; path: string; data: string }) {
  return apiPatch(`/api/doc_review/vfs/file`, payload);
}

export function sendChatMessage(fileId: string, body: ChatMessageRequest) {
  return apiPost<ChatMessageResponse>(`/api/doc_review/chat/${fileId}`, body);
}

export function getSocketToken() {
  return apiGet<{ token: string }>("/api/doc_review/token");
}
```

### **10.3 App Setup (`src/app`)**

#### **10.3.1 `queryClient.ts`**

Initializes the React Query client with default options.

```ts
// src/app/queryClient.ts
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});
```

#### **10.3.2 `router.tsx`**

Defines the application's routes using React Router.

```tsx
// src/app/router.tsx
import { createBrowserRouter } from "react-router-dom";
import AppLayout from "../components/layout/AppLayout";
import DocumentsPage from "../components/documents/DocumentsPage";
import WorkspacePage from "../components/workspace/WorkspacePage";
import TemplatesPage from "../components/templates/TemplatesPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { path: "/", element: <DocumentsPage /> },
      { path: "/documents", element: <DocumentsPage /> },
      { path: "/documents/:fileId", element: <WorkspacePage /> },
      { path: "/templates", element: <TemplatesPage /> },
    ],
  },
]);
```

### **10.4 Layout Components (`src/components/layout`)**

These components create the persistent shell of the application. `AppLayout.tsx` combines a `Sidebar` for navigation and a `TopBar` for global information, with the main content rendered via React Router's `<Outlet />`.

### **10.5 Documents List Components (`src/components/documents`)**

-   **`DocumentsPage.tsx`:** The main container for the documents list screen. It fetches the list of documents, manages filters, and controls the `UploadDialog`.
-   **`DocumentsTable.tsx`:** Renders the list of documents in a table, with each row linking to the corresponding workspace.
-   **`UploadDialog.tsx`:** A modal component that handles the two-step process of uploading a file and then registering it.

### **10.6 Workspace Components (`src/components/workspace`)**

This is the most complex part of the application, broken down by pane.

-   **`WorkspacePage.tsx`:** The top-level component that orchestrates the three panes. It fetches all necessary data for the given `fileId`, manages the WebSocket connection via `useDocReviewSocket`, and passes data and callbacks down to its children panes.

-   **Left Pane (`LeftPane/*`):**
    -   `LeftPaneTabs.tsx`: Manages the "Outline", "Issues", and "Artifacts" tabs.
    -   `OutlinePanel.tsx`: Renders a clickable document outline.
    -   `IssuesPanel.tsx`: Displays a filterable list of review issues.
    -   `ArtifactsPanel.tsx`: Shows an expandable tree of the document's VFS.

-   **Center Pane (`CenterPane/*`):**
    -   `CenterPane.tsx`: Manages the view mode (editor vs. diff).
    -   `EditorPane.tsx`: The core rich text editor component. It loads file content from the VFS, handles user edits, and auto-saves changes.
    -   `DiffView.tsx`: Fetches two file versions and displays a side-by-side diff.

-   **Right Pane (`RightPane/*`):**
    -   `RightPaneTabs.tsx`: Manages the "Chat" and "Activity" tabs.
    -   `ChatPanel.tsx`: Implements the chat interface for interacting with the agent.
    -   `ActivityPanel.tsx`: Displays the real-time stream of status and log events from the WebSocket.

### **10.7 Custom Hooks (`src/hooks`)**

-   **`useDocReviewSocket.ts`:** A reusable hook that manages the entire lifecycle of the WebSocket connection for a given `fileId`: connecting, joining the room, listening for events, and cleaning up on disconnect.
-   **`useEditorSelection.ts`:** Tracks the current text selection within the editor to provide context for chat messages.
-   **`useDocStatus.ts`:** A helper hook that computes a simplified status object from various API responses, used for displaying consistent status badges.

### **10.8 Type Definitions (`src/types`)**

**`docReview.ts`**
This file contains all necessary TypeScript interfaces for the API data structures, ensuring type safety throughout the application.

```ts
export interface DocumentRecord {
  file_id: string;
  name?: string;
  source_path?: string;
  created_at: string;
  updated_at: string;
  // ... and other fields from the backend
}

export interface VfsTreeItem {
  name: string;
  path: string;
  type: "file" | "directory";
}

export interface ChatMessageRequest {
  message: string;
  selected_text?: string;
}

// ... other type definitions
```

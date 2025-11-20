import "./log-forwarder";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import "./components/editor/lexical.css";

  console.log('%c🚀 Doc Review App v2.0 - SingleEditor Migration', 'color: #4CAF50; font-weight: bold; font-size: 14px;');
  console.log('%cFeatures: Character-based editing, Auto-save, Comments, AI Suggestions', 'color: #2196F3; font-size: 12px;');

  createRoot(document.getElementById("root")!).render(<App />);
  
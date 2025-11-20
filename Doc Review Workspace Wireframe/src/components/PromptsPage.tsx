import { useState, useEffect } from 'react';
import { Save, CheckCircle2, Sparkles, ChevronDown, Download } from 'lucide-react';

interface PromptConfig {
  id: string;
  title: string;
  description: string;
  filename: string;
}

const PROMPTS: PromptConfig[] = [
  {
    id: 'toc_review',
    title: '1. TOC & Structure Review',
    description: 'Analyzes document table of contents and overall structure',
    filename: 'phase1_toc_review.md',
  },
  {
    id: 'conceptual_coverage',
    title: '2. Conceptual Coverage',
    description: 'Evaluates completeness across universal policy domains',
    filename: 'phase2_check_conceptual_coverage.md',
  },
  {
    id: 'compliance_governance',
    title: '3. Compliance & Governance',
    description: 'Reviews regulatory precision and control strength',
    filename: 'phase2_check_compliance_governance.md',
  },
  {
    id: 'language_clarity',
    title: '4. Language & Clarity',
    description: 'Assesses writing quality, tone, and readability',
    filename: 'phase2_check_language_clarity.md',
  },
  {
    id: 'structural_presentation',
    title: '5. Structural & Presentation',
    description: 'Evaluates document flow and formatting',
    filename: 'phase2_check_structural_presentation.md',
  },
  {
    id: 'synthesis',
    title: '6. Synthesis Summary',
    description: 'Generates holistic assessment combining all checks',
    filename: 'phase2_synthesis_summary.md',
  },
];

export function PromptsPage() {
  const [selectedPrompt, setSelectedPrompt] = useState<PromptConfig>(PROMPTS[0]);
  const [promptContent, setPromptContent] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Load prompt when selection changes
  useEffect(() => {
    loadPrompt(selectedPrompt);
  }, [selectedPrompt]);

  const loadPrompt = async (prompt: PromptConfig) => {
    setIsLoading(true);
    setIsSaved(false);
    try {
      const response = await fetch(`/api/doc-review/prompts/${prompt.filename}`);
      if (response.ok) {
        const content = await response.text();
        setPromptContent(content);
      } else {
        setPromptContent(`# ${prompt.title}\n\nPrompt not found.`);
      }
    } catch (error) {
      console.error(`Failed to load prompt ${prompt.filename}:`, error);
      setPromptContent(`# ${prompt.title}\n\nError loading prompt.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const response = await fetch(`/api/doc-review/prompts/${selectedPrompt.filename}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: promptContent,
      });

      if (response.ok) {
        setIsSaved(true);
        setTimeout(() => setIsSaved(false), 3000);
      } else {
        alert(`Failed to save prompt: ${response.statusText}`);
      }
    } catch (error) {
      console.error(`Error saving prompt:`, error);
      alert('Error saving prompt');
    } finally {
      setIsSaving(false);
    }
  };

  const handlePromptChange = (value: string) => {
    setPromptContent(value);
    setIsSaved(false);
  };

  const handleDownload = () => {
    const blob = new Blob([promptContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedPrompt.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-slate-50 via-white to-slate-50 overflow-hidden">
      <div className="flex-1 flex flex-col max-w-7xl mx-auto w-full px-12 py-6 overflow-hidden">
        {/* Header */}
        <div className="mb-6 flex-shrink-0">
          <div className="flex items-center gap-3 mb-3">
            <div className="flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-gray-900">Analysis Prompts</h1>
              <p className="text-gray-500 mt-1">
                Customize AI evaluation criteria for your documents
              </p>
            </div>
          </div>
        </div>

        {/* Prompt Selector Card */}
        <div className="flex-1 flex flex-col bg-white rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
          {/* Dropdown Section */}
          <div className="bg-gradient-to-r from-gray-50 to-white p-6 border-b border-gray-200 flex-shrink-0">
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Select Prompt to Edit
            </label>
            <div className="relative">
              <select
                value={selectedPrompt.id}
                onChange={(e) => {
                  const prompt = PROMPTS.find(p => p.id === e.target.value);
                  if (prompt) setSelectedPrompt(prompt);
                }}
                className="w-full px-4 py-3 pr-10 text-base border-2 border-gray-300 rounded-xl bg-white text-gray-900 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all appearance-none cursor-pointer hover:border-gray-400"
              >
                {PROMPTS.map((prompt) => (
                  <option key={prompt.id} value={prompt.id}>
                    {prompt.title}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            </div>
            <p className="mt-3 text-sm text-gray-600 leading-relaxed">
              {selectedPrompt.description}
            </p>
          </div>

          {/* Textarea Section */}
          <div className="flex-1 flex flex-col p-6 overflow-hidden">
            <label className="block text-sm font-semibold text-gray-700 mb-3 flex-shrink-0">
              Prompt Content
            </label>
            {isLoading ? (
              <div className="flex-1 flex items-center justify-center bg-gray-50 rounded-xl border-2 border-dashed border-gray-300">
                <div className="text-center">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-3"></div>
                  <p className="text-gray-500 text-sm">Loading prompt...</p>
                </div>
              </div>
            ) : (
              <textarea
                value={promptContent}
                onChange={(e) => handlePromptChange(e.target.value)}
                className="flex-1 w-full px-5 py-4 border-2 border-gray-300 rounded-xl bg-white text-gray-900 font-mono text-[15px] leading-relaxed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none transition-all shadow-inner"
                placeholder="Enter prompt content..."
                spellCheck={false}
              />
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between p-6 pt-4 border-t border-gray-200 flex-shrink-0 bg-gray-50">
            <div className="flex items-center gap-3">
              <code className="px-3 py-1.5 bg-white text-gray-700 rounded-lg text-xs font-mono border border-gray-200 shadow-sm">
                {selectedPrompt.filename}
              </code>
              {isSaved && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-100 text-green-700 text-sm font-medium border border-green-200">
                  <CheckCircle2 className="w-4 h-4" />
                  Saved successfully
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleDownload}
                disabled={isLoading}
                className="inline-flex items-center gap-2 px-5 py-3 bg-white text-gray-700 border-2 border-gray-300 rounded-xl hover:bg-gray-50 hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-sm hover:shadow font-semibold"
                title="Download prompt as .md file"
              >
                <Download className="w-5 h-5" />
                Download
              </button>
              <button
                onClick={handleSave}
                disabled={isSaving || isLoading}
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl font-semibold"
              >
                <Save className="w-5 h-5" />
                {isSaving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

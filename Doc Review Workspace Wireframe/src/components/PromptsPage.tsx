import { useEffect, useState } from 'react';
import { Search, Save, RotateCcw } from 'lucide-react';
import { Button } from './ui/button';
import { listPrompts, getPrompt, updatePrompt, type ApiPrompt } from '@/lib/api';

export function PromptsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [prompts, setPrompts] = useState<ApiPrompt[]>([]);
  const [selectedPromptName, setSelectedPromptName] = useState<string | null>(null);
  const [promptContent, setPromptContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Load prompts list on mount
  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await listPrompts();
        setPrompts(res.prompts || []);
        if ((res.prompts || []).length > 0) {
          setSelectedPromptName(res.prompts[0].name);
        }
      } catch (e: any) {
        setError(e?.message || 'Failed to load prompts');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Load prompt content when selection changes
  useEffect(() => {
    async function loadPrompt() {
      if (!selectedPromptName) return;
      
      setLoading(true);
      setError(null);
      try {
        const res = await getPrompt(selectedPromptName);
        setPromptContent(res.content);
        setOriginalContent(res.content);
        setSaveSuccess(false);
      } catch (e: any) {
        setError(e?.message || 'Failed to load prompt');
        setPromptContent('');
        setOriginalContent('');
      } finally {
        setLoading(false);
      }
    }
    loadPrompt();
  }, [selectedPromptName]);

  const handleSave = async () => {
    if (!selectedPromptName) return;
    
    setSaving(true);
    setError(null);
    setSaveSuccess(false);
    
    try {
      await updatePrompt(selectedPromptName, promptContent);
      setOriginalContent(promptContent);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (e: any) {
      setError(e?.message || 'Failed to save prompt');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setPromptContent(originalContent);
    setSaveSuccess(false);
    setError(null);
  };

  const filteredPrompts = prompts.filter((p) =>
    p.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const hasChanges = promptContent !== originalContent;

  return (
    <div className="flex h-full bg-white">
      {/* Left Sidebar */}
      <div className="w-[260px] border-r border-neutral-200 flex flex-col">
        <div className="p-3 border-b border-neutral-200">
          <h2 className="text-sm font-semibold text-neutral-900 mb-3">Prompts</h2>
          
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
            <input
              type="text"
              placeholder="Search prompts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-3 py-2 border border-neutral-300 rounded focus:outline-none focus:ring-2 focus:ring-neutral-900 text-sm"
            />
          </div>
        </div>

        {/* Prompts List */}
        <div className="flex-1 overflow-y-auto">
          {loading && prompts.length === 0 && (
            <div className="px-4 py-2 text-sm text-neutral-600">Loading...</div>
          )}
          {error && prompts.length === 0 && (
            <div className="px-4 py-2 text-sm text-red-600">{error}</div>
          )}
          {!loading && !error && filteredPrompts.map((prompt) => (
            <button
              key={prompt.name}
              onClick={() => setSelectedPromptName(prompt.name)}
              className={`w-full text-left px-4 py-3 border-b border-neutral-100 hover:bg-neutral-50 transition-colors ${
                selectedPromptName === prompt.name ? 'bg-neutral-100' : ''
              }`}
            >
              <p className="text-neutral-900 mb-1 text-sm font-medium">{prompt.name}</p>
              <p className="text-neutral-500 text-xs">{prompt.filename}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-neutral-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-semibold text-neutral-900">
                {selectedPromptName || 'Select a prompt'}
              </h1>
              {selectedPromptName && (
                <p className="text-sm text-neutral-500 mt-1">
                  Edit system prompts used for document processing
                </p>
              )}
            </div>
            
            <div className="flex gap-2">
              {hasChanges && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  disabled={saving}
                >
                  <RotateCcw className="w-3 h-3 mr-2" />
                  Reset
                </Button>
              )}
              
              <Button
                variant="default"
                size="sm"
                onClick={handleSave}
                disabled={!selectedPromptName || !hasChanges || saving}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
              >
                {saving ? (
                  <>
                    <div className="w-3 h-3 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-3 h-3 mr-2" />
                    Save Changes
                  </>
                )}
              </Button>
            </div>
          </div>
          
          {/* Status Messages */}
          {saveSuccess && (
            <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-800">
              ✓ Prompt saved successfully
            </div>
          )}
          {error && prompts.length > 0 && (
            <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
              {error}
            </div>
          )}
        </div>

        {/* Editor */}
        <div className="flex-1 overflow-hidden">
          {selectedPromptName ? (
            <textarea
              value={promptContent}
              onChange={(e) => setPromptContent(e.target.value)}
              className="w-full h-full p-6 font-mono text-sm resize-none focus:outline-none border-0"
              placeholder="Enter prompt content..."
              style={{ fontFamily: "'Fira Code', 'Courier New', monospace" }}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-neutral-400">
              Select a prompt from the sidebar to begin editing
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


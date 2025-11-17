import { useEffect, useMemo, useState } from 'react';
import { Search, Upload } from 'lucide-react';
import { Button } from './ui/button';
import { TemplatePreviewModal } from './TemplatePreviewModal';
import { UploadModal } from './UploadModal';
import { getTemplate, listTemplates, type ApiTemplate } from '@/lib/api';

export function TemplatesPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [templates, setTemplates] = useState<ApiTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [selectedTemplateContent, setSelectedTemplateContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await listTemplates();
        setTemplates(res.templates || []);
        if ((res.templates || []).length > 0) {
          setSelectedTemplateId(res.templates[0].template_id);
        }
      } catch (e: any) {
        setError(e?.message || 'Failed to load templates');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    async function loadTemplate() {
      setSelectedTemplateContent(null);
      if (!selectedTemplateId) return;
      try {
        const res = await getTemplate(selectedTemplateId);
        if (res?.content) {
          setSelectedTemplateContent(JSON.stringify(res.content, null, 2));
        }
      } catch (e) {
        // Endpoint may require login; ignore for MVP
      }
    }
    loadTemplate();
  }, [selectedTemplateId]);

  const filteredTemplates = useMemo(() => {
    return templates.filter((t) =>
      t.template_id.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [templates, searchQuery]);

  return (
    <div className="flex h-full bg-white">
      {/* Left Sidebar */}
      <div className="w-[260px] border-r border-neutral-200 flex flex-col">
        {/* Header with Upload Button */}
        <div className="p-3 border-b border-neutral-200">
          <Button 
            size="sm" 
            onClick={() => setUploadModalOpen(true)}
            className="w-full mb-3"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Template
          </Button>
          
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
            <input
              type="text"
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-3 py-2 border border-neutral-300 rounded focus:outline-none focus:ring-2 focus:ring-neutral-900 text-sm"
            />
          </div>
        </div>

        {/* Templates List */}
        <div className="flex-1 overflow-y-auto">
          {loading && <div className="px-4 py-2 text-sm text-neutral-600">Loading…</div>}
          {error && <div className="px-4 py-2 text-sm text-red-600">{error}</div>}
          {!loading && !error && filteredTemplates.map((template) => (
            <button
              key={template.template_id}
              onClick={() => setSelectedTemplateId(template.template_id)}
              className={`w-full text-left px-4 py-3 border-b border-neutral-100 hover:bg-neutral-50 transition-colors ${
                selectedTemplateId === template.template_id ? 'bg-neutral-100' : ''
              }`}
            >
              <p className="text-neutral-900 mb-1 text-sm">{template.template_id}</p>
              <p className="text-neutral-600 text-xs">{template.location}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Right Preview Panel */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-8 py-6">
          {/* Header */}
          <div className="mb-5">
            <h1 className="text-neutral-900 mb-2">{selectedTemplateId || 'No template selected'}</h1>
            <p className="text-neutral-600 text-sm">
              {selectedTemplateId ? 'Template details' : 'Choose a template from the left'}
            </p>
          </div>

          {/* Preview */}
          <div className="mb-6 p-6 bg-neutral-50 border border-neutral-200 rounded">
            <div className="prose prose-neutral prose-sm max-w-none">
              {selectedTemplateId ? (
                selectedTemplateContent ? (
                  <pre className="text-xs text-neutral-800 whitespace-pre-wrap">{selectedTemplateContent}</pre>
                ) : (
                  <div className="text-sm text-neutral-600">
                    Preview requires login. You can still select this template when running reviews.
                  </div>
                )
              ) : (
                <div className="text-sm text-neutral-600">Select a template to preview details.</div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <Button onClick={() => setPreviewModalOpen(true)} size="sm" disabled={!selectedTemplateId}>
              Use this Template
            </Button>
            <Button variant="outline" size="sm" disabled={!selectedTemplateId}>
              View Source
            </Button>
          </div>
        </div>
      </div>

      {previewModalOpen && selectedTemplateId && (
        <TemplatePreviewModal
          template={{
            id: selectedTemplateId,
            name: selectedTemplateId,
            shortDescription: '',
            markdownContent: selectedTemplateContent || '',
          }}
          onClose={() => setPreviewModalOpen(false)}
        />
      )}
      
      {uploadModalOpen && (
        <UploadModal onClose={() => setUploadModalOpen(false)} />
      )}
    </div>
  );
}
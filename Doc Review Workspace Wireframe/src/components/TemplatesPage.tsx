import { useEffect, useMemo, useState } from 'react';
import { Search, Upload, Trash2 } from 'lucide-react';
import { Button } from './ui/button';
import { TemplatePreviewModal } from './TemplatePreviewModal';
import { UploadTemplateModal } from './UploadTemplateModal';
import { getTemplate, listTemplates, getTemplateContent, deleteTemplate, type ApiTemplate } from '@/lib/api';

export function TemplatesPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [templates, setTemplates] = useState<ApiTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [selectedTemplateContent, setSelectedTemplateContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await listTemplates();
        console.log('[TemplatesPage] Raw API response:', res);
        // Handle both string array and object array formats
        const templateList = (res.templates || []).map((t: any) => {
          if (typeof t === 'string') {
            return { template_id: t, path: '', size: 0, location: 'data/templates' };
          }
          return t;
        });
        console.log('[TemplatesPage] Processed templates:', templateList);
        setTemplates(templateList);
        if (templateList.length > 0) {
          setSelectedTemplateId(templateList[0].template_id);
        }
      } catch (e: any) {
        setError(e?.message || 'Failed to load templates');
        console.error('[TemplatesPage] Load error:', e);
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
        const res = await getTemplateContent(selectedTemplateId);
        if (res?.content) {
          setSelectedTemplateContent(res.content);
        }
      } catch (e) {
        console.error('Failed to load template:', e);
        setSelectedTemplateContent(null);
      }
    }
    loadTemplate();
  }, [selectedTemplateId]);

  const handleDelete = async (templateId: string) => {
    if (!confirm(`Are you sure you want to delete template "${templateId}"?`)) {
      return;
    }
    
    setDeletingId(templateId);
    try {
      await deleteTemplate(templateId);
      // Refresh the list
      const res = await listTemplates();
      const templateList = (res.templates || []).map((t: any) => {
        if (typeof t === 'string') {
          return { template_id: t, path: '', size: 0, location: '' };
        }
        return t;
      });
      setTemplates(templateList);
      
      // If deleted template was selected, clear selection
      if (selectedTemplateId === templateId) {
        setSelectedTemplateId(templateList.length > 0 ? templateList[0].template_id : null);
      }
    } catch (e: any) {
      alert(`Failed to delete: ${e?.message || 'Unknown error'}`);
    } finally {
      setDeletingId(null);
    }
  };

  const filteredTemplates = useMemo(() => {
    return templates.filter((t) =>
      t?.template_id?.toLowerCase().includes(searchQuery.toLowerCase())
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
            <div
              key={template.template_id}
              className={`w-full text-left px-4 py-3 border-b border-neutral-100 hover:bg-neutral-50 transition-colors flex items-center justify-between group ${
                selectedTemplateId === template.template_id ? 'bg-neutral-100' : ''
              }`}
            >
              <button
                onClick={() => setSelectedTemplateId(template.template_id)}
                className="flex-1 text-left"
              >
                <p className="text-neutral-900 mb-1 text-sm">{template.template_id}</p>
                <p className="text-neutral-600 text-xs">{template.location}</p>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(template.template_id);
                }}
                disabled={deletingId === template.template_id}
                className="ml-2 p-2 text-neutral-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors opacity-0 group-hover:opacity-100 disabled:opacity-50"
                title="Delete template"
              >
                {deletingId === template.template_id ? (
                  <div className="w-4 h-4 border-2 border-neutral-300 border-t-neutral-600 rounded-full animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
              </button>
            </div>
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
          <div className="mb-6 p-6 bg-white border border-neutral-200 rounded-lg shadow-sm">
            <div className="prose prose-neutral prose-sm max-w-none">
              {selectedTemplateId ? (
                selectedTemplateContent ? (
                  <div className="markdown-preview">
                    <pre className="whitespace-pre-wrap font-mono text-sm text-neutral-800 leading-relaxed">
                      {selectedTemplateContent}
                    </pre>
                  </div>
                ) : (
                  <div className="flex items-center justify-center py-12">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neutral-900 mx-auto mb-3"></div>
                      <div className="text-sm text-neutral-600">Loading template...</div>
                    </div>
                  </div>
                )
              ) : (
                <div className="text-sm text-neutral-500 text-center py-12">
                  Select a template from the left to preview its content
                </div>
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
        <UploadTemplateModal 
          onClose={() => setUploadModalOpen(false)} 
          onSuccess={async () => {
            console.log('[TemplatesPage] Upload success, reloading list...');
            // Wait for backend to save file
            await new Promise(resolve => setTimeout(resolve, 500));
            // Reload templates list after successful upload
            setLoading(true);
            setError(null);
            try {
              const res = await listTemplates();
              console.log('[TemplatesPage] Reloaded templates after upload:', res);
              const templateList = (res.templates || []).map((t: any) => {
                if (typeof t === 'string') {
                  return { template_id: t, path: '', size: 0, location: 'data/templates' };
                }
                return t;
              });
              console.log('[TemplatesPage] Setting templates state to:', templateList);
              setTemplates(templateList);
              // Select the newly uploaded template if it's the only change
              if (templateList.length > templates.length && templateList.length > 0) {
                const newTemplate = templateList.find(t => !templates.some(old => old.template_id === t.template_id));
                if (newTemplate) {
                  console.log('[TemplatesPage] Auto-selecting new template:', newTemplate.template_id);
                  setSelectedTemplateId(newTemplate.template_id);
                }
              }
            } catch (err: any) {
              console.error('[TemplatesPage] Failed to reload templates:', err);
            } finally {
              setLoading(false);
            }
          }}
        />
      )}
    </div>
  );
}
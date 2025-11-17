import { X, ChevronDown, ChevronRight } from 'lucide-react';
import { Button } from './ui/button';
import { useState } from 'react';

interface Template {
  id: string;
  name: string;
  shortDescription: string;
  markdownContent: string;
}

interface TemplatePreviewModalProps {
  template: Template;
  onClose: () => void;
}

export function TemplatePreviewModal({ template, onClose }: TemplatePreviewModalProps) {
  const [markdownExpanded, setMarkdownExpanded] = useState(true);

  const handleUseTemplate = () => {
    console.log('Using template:', template.id);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-neutral-200 flex-shrink-0">
          <h2 className="text-neutral-900">{template.name}</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-neutral-100 rounded transition-colors"
          >
            <X className="w-5 h-5 text-neutral-500" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-5 overflow-y-auto flex-1">
          {/* Description */}
          <div>
            <h3 className="text-neutral-900 mb-2 text-sm">Description</h3>
            <p className="text-neutral-700 text-sm">{template.shortDescription}</p>
          </div>

          {/* Markdown Content (Collapsible) */}
          <div>
            <button
              onClick={() => setMarkdownExpanded(!markdownExpanded)}
              className="flex items-center gap-2 w-full text-left mb-2 hover:text-neutral-900 transition-colors"
            >
              {markdownExpanded ? (
                <ChevronDown className="w-4 h-4 text-neutral-600" />
              ) : (
                <ChevronRight className="w-4 h-4 text-neutral-600" />
              )}
              <h3 className="text-neutral-900 text-sm">Template Content</h3>
            </button>
            
            {markdownExpanded && (
              <div className="bg-neutral-50 border border-neutral-200 p-4 rounded overflow-x-auto">
                <pre className="text-xs text-neutral-700 whitespace-pre-wrap font-mono">
                  {template.markdownContent}
                </pre>
              </div>
            )}
          </div>

          {/* Preview Rendered */}
          <div>
            <h3 className="text-neutral-900 mb-2 text-sm">Preview</h3>
            <div className="bg-white border border-neutral-200 p-4 rounded">
              <div className="prose prose-neutral prose-sm max-w-none">
                {template.markdownContent.split('\n').map((line, index) => {
                  if (line.startsWith('# ')) {
                    return <h1 key={index} className="text-base mt-3 mb-2">{line.replace('# ', '')}</h1>;
                  } else if (line.startsWith('## ')) {
                    return <h2 key={index} className="text-sm mt-2 mb-1">{line.replace('## ', '')}</h2>;
                  } else if (line.startsWith('### ')) {
                    return <h3 key={index} className="text-xs mt-2 mb-1">{line.replace('### ', '')}</h3>;
                  } else if (line.startsWith('- ')) {
                    return <li key={index} className="ml-4 text-xs">{line.replace('- ', '')}</li>;
                  } else if (line.startsWith('|')) {
                    return <div key={index} className="text-xs text-neutral-600 font-mono">{line}</div>;
                  } else if (line.trim() === '') {
                    return <br key={index} />;
                  } else {
                    return <p key={index} className="text-xs text-neutral-700 mb-1">{line}</p>;
                  }
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-3 border-t border-neutral-200 bg-neutral-50 flex-shrink-0">
          <Button variant="outline" onClick={onClose} size="sm">
            Cancel
          </Button>
          <Button onClick={handleUseTemplate} size="sm">
            Use Template
          </Button>
        </div>
      </div>
    </div>
  );
}

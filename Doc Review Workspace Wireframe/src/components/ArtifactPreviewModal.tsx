import { X } from 'lucide-react';
import { Button } from './ui/button';

interface ArtifactPreviewModalProps {
  fileName: string;
  filePath: string;
  content: string;
  fileType: 'json' | 'markdown' | 'text';
  onClose: () => void;
}

export function ArtifactPreviewModal({
  fileName,
  filePath,
  content,
  fileType,
  onClose,
}: ArtifactPreviewModalProps) {
  const handleOpenInEditor = () => {
    console.log('Opening in editor:', filePath);
    onClose();
  };

  const renderContent = () => {
    if (fileType === 'json') {
      try {
        const parsed = JSON.parse(content);
        return (
          <pre className="text-sm text-neutral-100">
            {JSON.stringify(parsed, null, 2)}
          </pre>
        );
      } catch {
        return <pre className="text-sm text-neutral-100">{content}</pre>;
      }
    }

    if (fileType === 'markdown') {
      return (
        <div className="prose prose-invert max-w-none">
          <pre className="text-sm text-neutral-100 whitespace-pre-wrap">{content}</pre>
        </div>
      );
    }

    return <pre className="text-sm text-neutral-100 whitespace-pre-wrap font-mono">{content}</pre>;
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-200 flex-shrink-0">
          <div>
            <h2 className="text-neutral-900">{fileName}</h2>
            <p className="text-neutral-500">{filePath}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-neutral-100 rounded transition-colors"
          >
            <X className="w-5 h-5 text-neutral-500" />
          </button>
        </div>

        {/* Body - Scrollable Content */}
        <div className="flex-1 overflow-auto bg-neutral-900 p-6">
          {renderContent()}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-neutral-200 bg-neutral-50 flex-shrink-0">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button onClick={handleOpenInEditor}>
            Open in Editor Pane
          </Button>
        </div>
      </div>
    </div>
  );
}

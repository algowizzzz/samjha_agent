import { AlertCircle, LayoutTemplate, FolderOpen, RotateCw, FileText } from 'lucide-react';
import { Button } from './ui/button';

interface PhaseFailedProps {
  phaseName: string;
  errorMessage: string;
  onRetry: () => void;
  onViewLogs: () => void;
}

export function PhaseFailed({ phaseName, errorMessage, onRetry, onViewLogs }: PhaseFailedProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 bg-white">
      <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center mb-4">
        <AlertCircle className="w-8 h-8 text-red-600" />
      </div>
      <h2 className="text-neutral-900 mb-2">{phaseName} Failed</h2>
      <p className="text-neutral-600 mb-6 text-center max-w-md">
        {errorMessage}
      </p>
      <div className="flex gap-3">
        <Button variant="outline" onClick={onViewLogs}>
          View Logs
        </Button>
        <Button onClick={onRetry}>
          <RotateCw className="w-4 h-4 mr-2" />
          Retry
        </Button>
      </div>
    </div>
  );
}

interface NoTemplatesProps {
  onAddTemplate: () => void;
}

export function NoTemplates({ onAddTemplate }: NoTemplatesProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 bg-white">
      <LayoutTemplate className="w-16 h-16 text-neutral-300 mb-4" />
      <h2 className="text-neutral-900 mb-2">No templates found</h2>
      <p className="text-neutral-600 mb-6 text-center max-w-md">
        Create your first template to get started with document reviews
      </p>
      <Button onClick={onAddTemplate}>
        Add Template
      </Button>
    </div>
  );
}

export function NoArtifacts() {
  return (
    <div className="flex flex-col items-center justify-center p-12 bg-neutral-50 border border-neutral-200 rounded-lg m-4">
      <FolderOpen className="w-12 h-12 text-neutral-300 mb-3" />
      <p className="text-neutral-600 text-center">
        Artifacts will appear after Phase 1 completes
      </p>
    </div>
  );
}

export function NoDocuments() {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 bg-white">
      <FileText className="w-16 h-16 text-neutral-300 mb-4" />
      <h2 className="text-neutral-900 mb-2">No documents yet</h2>
      <p className="text-neutral-600 mb-6 text-center max-w-md">
        Upload your first document to begin the review process
      </p>
    </div>
  );
}

interface NoResultsProps {
  message?: string;
}

export function NoResults({ message = 'No results found' }: NoResultsProps) {
  return (
    <div className="flex flex-col items-center justify-center p-12">
      <p className="text-neutral-600 text-center">{message}</p>
    </div>
  );
}

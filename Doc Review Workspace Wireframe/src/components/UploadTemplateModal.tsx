import { useState } from 'react';
import { X, Upload, File } from 'lucide-react';
import { Button } from './ui/button';
import { uploadTemplate } from '@/lib/api';

interface UploadTemplateModalProps {
  onClose: () => void;
  onSuccess?: () => void;
}

export function UploadTemplateModal({ onClose, onSuccess }: UploadTemplateModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.md')) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError('Only .md (Markdown) files are supported');
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.name.endsWith('.md')) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError('Only .md (Markdown) files are supported');
      }
    }
  };

  const handleUpload = async () => {
    setError(null);
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }
    if (!selectedFile.name.endsWith('.md')) {
      setError('Only .md (Markdown) files are supported');
      return;
    }

    setSubmitting(true);
    try {
      const result = await uploadTemplate(selectedFile);
      // eslint-disable-next-line no-console
      console.debug('[UploadTemplateModal] Template uploaded successfully', result);
      onSuccess?.();
      onClose();
    } catch (e: any) {
      // eslint-disable-next-line no-console
      console.error('[UploadTemplateModal] Upload failed:', e);
      setError(e?.message || 'Template upload failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-neutral-200">
          <h2 className="text-neutral-900">Upload Template</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-neutral-100 rounded transition-colors"
          >
            <X className="w-5 h-5 text-neutral-500" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-5">
          {error && (
            <div className="text-sm text-red-600 bg-red-50 p-3 rounded">{error}</div>
          )}
          
          {/* File Upload Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
              isDragging
                ? 'border-blue-500 bg-blue-50'
                : 'border-neutral-300 bg-neutral-50'
            }`}
          >
            {selectedFile ? (
              <div className="flex items-center justify-center gap-3">
                <File className="w-8 h-8 text-neutral-600" />
                <div className="text-left">
                  <p className="text-neutral-900 text-sm">{selectedFile.name}</p>
                  <p className="text-neutral-500 text-xs">
                    {(selectedFile.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              </div>
            ) : (
              <>
                <Upload className="w-10 h-10 text-neutral-400 mx-auto mb-3" />
                <p className="text-neutral-700 mb-2 text-sm">
                  Drag & drop a markdown (.md) file here or browse
                </p>
                <label className="inline-block">
                  <input
                    type="file"
                    onChange={handleFileSelect}
                    className="hidden"
                    accept=".md"
                  />
                  <span className="text-blue-600 hover:text-blue-700 cursor-pointer text-sm">
                    Choose file
                  </span>
                </label>
              </>
            )}
          </div>

          <div className="text-xs text-neutral-600 bg-neutral-50 p-3 rounded">
            <p className="font-medium mb-1">Note:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Only Markdown (.md) files are supported</li>
              <li>Template will be saved with the filename (without .md extension)</li>
              <li>Template can be used for document review gap analysis</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-3 border-t border-neutral-200 bg-neutral-50">
          <Button variant="outline" onClick={onClose} size="sm">
            Cancel
          </Button>
          <Button onClick={handleUpload} size="sm" disabled={submitting || !selectedFile}>
            {submitting ? 'Uploading…' : 'Upload Template'}
          </Button>
        </div>
      </div>
    </div>
  );
}


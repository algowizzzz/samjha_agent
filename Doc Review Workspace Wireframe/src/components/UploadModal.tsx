import { useState } from 'react';
import { X, Upload, File } from 'lucide-react';
import { Button } from './ui/button';
import { uploadFile, registerDocument } from '@/lib/api';

interface UploadModalProps {
  onClose: () => void;
}

export function UploadModal({ onClose }: UploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentTitle, setDocumentTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    // eslint-disable-next-line no-console
    console.debug('[UI] Drag over upload dropzone');
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      // eslint-disable-next-line no-console
      console.debug('[UI] File dropped', e.dataTransfer.files[0].name);
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      // eslint-disable-next-line no-console
      console.debug('[UI] File selected', e.target.files[0].name);
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    setError(null);
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }
    // eslint-disable-next-line no-console
    console.debug('[UI] Click Upload & Register', { name: selectedFile.name, size: selectedFile.size });
    setSubmitting(true);
    try {
      const uploadRes = await uploadFile(selectedFile);
      // eslint-disable-next-line no-console
      console.debug('[UploadModal] uploadFile result', uploadRes);
      // Try to register; if file_id collision (409), retry with suffix
      try {
        await registerDocument({
          source_path: uploadRes.saved_path,
          file_id: uploadRes.file_id,
        });
      } catch (e: any) {
        // eslint-disable-next-line no-console
        console.error('registerDocument failed:', e?.message);
        const needsRetry = typeof e?.message === 'string' && e.message.toLowerCase().includes('already exists');
        if (!needsRetry) throw e;
        const suffix = Math.random().toString(36).slice(2, 8);
        await registerDocument({
          source_path: uploadRes.saved_path,
          file_id: `${uploadRes.file_id}-${suffix}`,
        });
      }
      // eslint-disable-next-line no-console
      console.debug('[UploadModal] registerDocument success');
      onClose();
    } catch (e: any) {
      setError(e?.message || 'Upload failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-neutral-200">
          <h2 className="text-neutral-900">Upload & Review Document</h2>
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
            <div className="text-sm text-red-600">{error}</div>
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
                  Drag & drop file here or browse
                </p>
                <label className="inline-block">
                  <input
                    type="file"
                    onChange={handleFileSelect}
                    className="hidden"
                    accept=".doc,.docx,.pdf,.txt,.md"
                  />
                  <span className="text-blue-600 hover:text-blue-700 cursor-pointer text-sm">
                    Choose file
                  </span>
                </label>
              </>
            )}
          </div>

          {/* Optional Fields */}
          <div className="space-y-4">
            <div>
              <label className="block text-neutral-700 mb-1.5 text-sm">
                Document Title (optional)
              </label>
              <input
                type="text"
                value={documentTitle}
                onChange={(e) => setDocumentTitle(e.target.value)}
                placeholder="Enter document title"
                className="w-full px-3 py-2 border border-neutral-300 rounded focus:outline-none focus:ring-2 focus:ring-neutral-900 text-sm"
              />
            </div>

            <div>
              <label className="block text-neutral-700 mb-1.5 text-sm">
                Description (optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter description"
                rows={3}
                className="w-full px-3 py-2 border border-neutral-300 rounded focus:outline-none focus:ring-2 focus:ring-neutral-900 text-sm"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-3 border-t border-neutral-200 bg-neutral-50">
          <Button variant="outline" onClick={onClose} size="sm">
            Cancel
          </Button>
          <Button onClick={handleUpload} size="sm" disabled={submitting}>
            {submitting ? 'Uploading…' : 'Upload & Register'}
          </Button>
        </div>
      </div>
    </div>
  );
}

import { useState } from 'react';
import { X, Upload, File } from 'lucide-react';
import { Button } from './ui/button';
import { uploadFile, registerDocument, runIngestion } from '@/lib/api';
import { UploadProgress } from './UploadProgress';

interface UploadModalProps {
  onClose: () => void;
}

type ProcessStep = {
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  message?: string;
};

export function UploadModal({ onClose }: UploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentTitle, setDocumentTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processSteps, setProcessSteps] = useState<ProcessStep[]>([]);
  const [useDirectJSON, setUseDirectJSON] = useState(true); // Direct PDF→JSON vs Markdown→JSON
  const [uploadAsTemplate, setUploadAsTemplate] = useState(false); // Simple MD upload for templates

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

  const updateStep = (index: number, status: ProcessStep['status'], message?: string) => {
    setProcessSteps(prev => prev.map((step, i) => 
      i === index ? { ...step, status, message } : step
    ));
  };

  const handleUpload = async () => {
    setError(null);
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }
    
    // Initialize progress steps
    const steps: ProcessStep[] = uploadAsTemplate 
      ? [
          { name: 'Uploading file', status: 'pending' },
          { name: 'Registering document', status: 'pending' },
          { name: 'Saving as template', status: 'pending' },
        ]
      : useDirectJSON
      ? [
          { name: 'Uploading file', status: 'pending' },
          { name: 'Registering document', status: 'pending' },
          { name: 'Converting PDF to images', status: 'pending' },
          { name: 'Extracting structure & formatting (Vision AI)', status: 'pending' },
          { name: 'Creating semantic blocks', status: 'pending' },
          { name: 'Finalizing document', status: 'pending' },
        ]
      : [
          { name: 'Uploading file', status: 'pending' },
          { name: 'Registering document', status: 'pending' },
          { name: 'Converting to markdown', status: 'pending' },
          { name: 'Analyzing structure', status: 'pending' },
          { name: 'Creating semantic blocks', status: 'pending' },
          { name: 'Verifying content', status: 'pending' },
          { name: 'Finalizing document', status: 'pending' },
        ];
    setProcessSteps(steps);
    setSubmitting(true);
    
    try {
      // Step 1: Upload file
      updateStep(0, 'in_progress', 'Uploading...');
      const uploadRes = await uploadFile(selectedFile);
      updateStep(0, 'completed', `Uploaded ${selectedFile.name}`);
      
      // Step 2: Register document
      updateStep(1, 'in_progress', 'Registering...');
      let fileId = uploadRes.file_id;
      try {
        await registerDocument({
          source_path: uploadRes.saved_path,
          file_id: fileId,
        });
      } catch (e: any) {
        const needsRetry = typeof e?.message === 'string' && e.message.toLowerCase().includes('already exists');
        if (needsRetry) {
          const suffix = Math.random().toString(36).slice(2, 8);
          fileId = `${uploadRes.file_id}-${suffix}`;
          await registerDocument({
            source_path: uploadRes.saved_path,
            file_id: fileId,
          });
        } else {
          throw e;
        }
      }
      updateStep(1, 'completed', `Document ID: ${fileId}`);
      
      // Step 3-7: Run full ingestion (converts PDF → markdown → blocks)
      updateStep(2, 'in_progress', uploadAsTemplate ? 'Simple upload...' : 'Processing PDF pages...');
      let ingestionResult;
      if (!uploadAsTemplate) {
        ingestionResult = await runIngestion(fileId, { useDirectJSON });
      }
      
      // Update all remaining steps as completed
      if (uploadAsTemplate) {
        updateStep(2, 'completed', 'Template saved');
      } else if (useDirectJSON) {
        updateStep(2, 'completed', 'PDF converted to images');
        updateStep(3, 'completed', 'Structure & formatting extracted');
        updateStep(4, 'completed', 'Semantic blocks created');
        updateStep(5, 'completed', 'Document ready!');
      } else {
        updateStep(2, 'completed', 'Markdown generated');
        updateStep(3, 'completed', 'Structure analyzed');
        updateStep(4, 'completed', 'Semantic blocks created');
        updateStep(5, 'completed', 'Content verified');
        updateStep(6, 'completed', 'Document ready!');
      }
      
      // Wait a moment to show completion
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      onClose();
    } catch (e: any) {
      // Mark current step as error
      const currentStepIndex = steps.findIndex(s => s.status === 'in_progress');
      if (currentStepIndex >= 0) {
        updateStep(currentStepIndex, 'error', e?.message || 'Failed');
      }
      setError(e?.message || 'Processing failed');
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
            <div className="text-sm text-red-600 bg-red-50 p-3 rounded">{error}</div>
          )}
          
          {/* Show progress if processing */}
          {submitting && processSteps.length > 0 ? (
            <div className="py-4">
              <UploadProgress steps={processSteps} />
            </div>
          ) : (
            <>
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

              {/* Processing Options */}
              {!submitting && (
                <div className="space-y-3 pt-2 border-t border-neutral-200">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="uploadAsTemplate"
                      checked={uploadAsTemplate}
                      onChange={(e) => {
                        setUploadAsTemplate(e.target.checked);
                        if (e.target.checked) {
                          setUseDirectJSON(false); // Template mode uses simple MD upload
                        }
                      }}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="uploadAsTemplate" className="text-sm text-neutral-700">
                      Upload as template (simple MD, no processing)
                    </label>
                  </div>

                  {!uploadAsTemplate && (
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="useDirectJSON"
                        checked={useDirectJSON}
                        onChange={(e) => setUseDirectJSON(e.target.checked)}
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="useDirectJSON" className="text-sm text-neutral-700">
                        🚀 Use direct PDF→JSON (better formatting detection)
                      </label>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-3 border-t border-neutral-200 bg-neutral-50">
          <Button variant="outline" onClick={onClose} size="sm" disabled={submitting}>
            {submitting ? 'Processing...' : 'Cancel'}
          </Button>
          <Button onClick={handleUpload} size="sm" disabled={submitting || !selectedFile}>
            {submitting ? 'Processing…' : uploadAsTemplate ? 'Upload Template' : 'Upload & Process'}
          </Button>
        </div>
      </div>
    </div>
  );
}

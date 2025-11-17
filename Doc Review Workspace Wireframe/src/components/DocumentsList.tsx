import { useEffect, useMemo, useState } from 'react';
import { Search, Upload, FileText, GripVertical } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { UploadModal } from './UploadModal';
import { listDocuments, type ApiDocument } from '@/lib/api';

type ColumnKey = 'name' | 'fileId' | 'source' | 'status' | 'lastUpdated';

interface Column {
  key: ColumnKey;
  label: string;
}

const defaultColumns: Column[] = [
  { key: 'name', label: 'Document Name' },
  { key: 'fileId', label: 'File ID' },
  { key: 'source', label: 'Source' },
  { key: 'status', label: 'Status' },
  { key: 'lastUpdated', label: 'Last Updated' },
];

interface DocumentsListProps {
  onOpenDocument: (fileId: string) => void;
}

export function DocumentsList({ onOpenDocument }: DocumentsListProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [columns, setColumns] = useState<Column[]>(defaultColumns);
  const [draggedColumn, setDraggedColumn] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [docs, setDocs] = useState<ApiDocument[]>([]);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const res = await listDocuments();
      // eslint-disable-next-line no-console
      console.debug('[DocumentsList] listDocuments ->', res);
      setDocs(res.documents || []);
    } catch (e: any) {
      // eslint-disable-next-line no-console
      console.error('[DocumentsList] listDocuments error:', e);
      setError(e?.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  const getStatusBadge = (status: string | undefined) => {
    if (status === 'completed' || status === 'ready') {
      return <Badge className="bg-emerald-100 text-emerald-800 text-xs">{status}</Badge>;
    }
    if (status === 'running') {
      return <Badge className="bg-blue-100 text-blue-800 text-xs">running</Badge>;
    }
    if (status === 'error') {
      return <Badge className="bg-red-100 text-red-800 text-xs">error</Badge>;
    }
    return <Badge variant="outline" className="bg-neutral-50 text-neutral-600 text-xs">{status || 'unknown'}</Badge>;
  };

  const filteredDocuments = useMemo(() => {
    const normalized = docs.map(d => {
      const name = d.file_metadata?.['name'] as string | undefined;
      return {
        ...d,
        _displayName: name || d.file_id,
      };
    });
    return normalized.filter((doc) => {
      const matchesSearch =
        (doc._displayName || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        (doc.file_id || '').toLowerCase().includes(searchQuery.toLowerCase());
      if (!matchesSearch) return false;
      if (statusFilter === 'All') return true;
      if (statusFilter === 'In Progress') return doc.status === 'running';
      if (statusFilter === 'Completed') return doc.status === 'completed' || doc.status === 'ready';
      if (statusFilter === 'Needs Review') return doc.status === 'unknown' || !doc.status;
      return true;
    });
  }, [docs, searchQuery, statusFilter]);

  const handleDragStart = (index: number) => {
    setDraggedColumn(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (draggedColumn === null || draggedColumn === index) return;
    
    const newColumns = [...columns];
    const draggedItem = newColumns[draggedColumn];
    newColumns.splice(draggedColumn, 1);
    newColumns.splice(index, 0, draggedItem);
    
    setColumns(newColumns);
    setDraggedColumn(index);
  };

  const handleDragEnd = () => {
    setDraggedColumn(null);
  };

  const renderCellContent = (doc: Document, columnKey: ColumnKey) => {
    switch (columnKey) {
      case 'name':
        // @ts-ignore
        return <span className="text-neutral-900">{(doc as any)._displayName}</span>;
      case 'fileId':
        // @ts-ignore
        return <span className="text-neutral-600">{(doc as any).file_id}</span>;
      case 'source':
        // @ts-ignore
        return <span className="text-neutral-600">{(doc as any).source_path?.split('/').pop() || 'Upload'}</span>;
      case 'status':
        // @ts-ignore
        return getStatusBadge((doc as any).status);
      case 'lastUpdated':
        // @ts-ignore
        return <span className="text-neutral-600">{(doc as any).updated_at || ''}</span>;
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-white">
        <FileText className="w-16 h-16 text-neutral-300 mb-4" />
        <h2 className="text-neutral-900 mb-2">Loading documents…</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-white">
        <FileText className="w-16 h-16 text-red-300 mb-4" />
        <h2 className="text-neutral-900 mb-2">Failed to load</h2>
        <p className="text-neutral-600 mb-6 text-sm">{error}</p>
        <Button onClick={() => refresh()}>Retry</Button>
      </div>
    );
  }

  if (filteredDocuments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-white">
        <FileText className="w-16 h-16 text-neutral-300 mb-4" />
        <h2 className="text-neutral-900 mb-2">No documents yet</h2>
        <p className="text-neutral-600 mb-6">Upload your first document to get started</p>
        <Button onClick={() => setUploadModalOpen(true)}>
          <Upload className="w-4 h-4 mr-2" />
          Upload & Review
        </Button>
        
        {uploadModalOpen && (
          <UploadModal onClose={() => { setUploadModalOpen(false); refresh(); }} />
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="border-b border-neutral-200 px-6 py-3">
        <div className="flex items-center justify-between mb-3">
          <h1 className="text-neutral-900">Documents</h1>
          <Button size="sm" onClick={() => { console.debug('[UI] Click Upload & Review'); setUploadModalOpen(true); }}>
            <Upload className="w-4 h-4 mr-2" />
            Upload & Review
          </Button>
        </div>

        {/* Filters */}
        <div className="flex gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-neutral-300 rounded focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent text-sm"
            />
          </div>
          
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border border-neutral-300 rounded bg-white text-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-900 text-sm"
          >
            <option value="All">All</option>
            <option value="Needs Review">Needs Review</option>
            <option value="In Progress">In Progress</option>
            <option value="Completed">Completed</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="bg-neutral-50 border-b border-neutral-200 sticky top-0">
            <tr>
              {columns.map((column, index) => (
                <th
                  key={column.key}
                  draggable
                  onDragStart={() => handleDragStart(index)}
                  onDragOver={(e) => handleDragOver(e, index)}
                  onDragEnd={handleDragEnd}
                  className="text-left px-6 py-2 text-neutral-700 text-sm cursor-move hover:bg-neutral-100 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <GripVertical className="w-3 h-3 text-neutral-400" />
                    {column.label}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredDocuments.map((doc: any) => (
              <tr
                key={doc.file_id}
                onClick={() => { console.debug('[UI] Click row open document', doc.file_id); onOpenDocument(doc.file_id); }}
                className="border-b border-neutral-100 hover:bg-neutral-50 cursor-pointer transition-colors"
              >
                {columns.map((column) => (
                  <td key={column.key} className="px-6 py-3">
                    {renderCellContent(doc, column.key)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {uploadModalOpen && (
        <UploadModal onClose={() => { setUploadModalOpen(false); refresh(); }} />
      )}
    </div>
  );
}

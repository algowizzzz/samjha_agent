import { useEffect, useState } from 'react';
import { BlockMetadata } from '@/lib/api';
import * as Diff from 'diff-match-patch';
import { AlertCircle, FileText, ChevronDown, ChevronRight, ChevronUp } from 'lucide-react';

interface DiffViewProps {
  blocks: Array<{
    id: string;
    type: string;
    content: string;
    changeHistory: Array<{
      timestamp: string;
      type: string;
      original: string;
      modified: string;
      reason?: string;
      user?: string;
    }>;
  }>;
  blockMetadata?: BlockMetadata[];
}

interface WordDiff {
  type: 'added' | 'removed' | 'unchanged';
  text: string;
}

interface BlockDiff {
  blockId: string;
  blockNum: number;
  blockType: string;
  hasChanges: boolean;
  original: string;
  current: string;
  wordDiffs: WordDiff[];
  reason?: string;
}

export function DiffView({ blocks, blockMetadata }: DiffViewProps) {
  const [blockDiffs, setBlockDiffs] = useState<BlockDiff[]>([]);
  const [expandedBlocks, setExpandedBlocks] = useState<Set<string>>(new Set());
  const [currentChangeIndex, setCurrentChangeIndex] = useState<number>(0);

  useEffect(() => {
    computeBlockDiffs();
  }, [blocks]);

  const computeBlockDiffs = () => {
    const diffs: BlockDiff[] = [];

    blocks.forEach((block, index) => {
      // Find the original content from change history
      let original = block.content;
      let reason: string | undefined;

      if (block.changeHistory && block.changeHistory.length > 0) {
        const firstChange = block.changeHistory.find(
          h => h.type === 'ai_suggested' || h.type === 'verified' || h.type === 'ai_applied'
        );
        if (firstChange) {
          original = firstChange.original;
          reason = firstChange.reason;
        }
      }

      const current = block.content;
      const hasChanges = original !== current;

      const metadata = blockMetadata?.find(m => m.id === block.id);
      const blockNum = metadata?.block_num || index + 1;

      // Compute word-level diff
      const wordDiffs = computeWordDiff(original, current);

      diffs.push({
        blockId: block.id,
        blockNum,
        blockType: block.type,
        hasChanges,
        original,
        current,
        wordDiffs,
        reason
      });
    });

    setBlockDiffs(diffs);
    
    // Auto-expand first changed block
    const firstChanged = diffs.find(d => d.hasChanges);
    if (firstChanged) {
      setExpandedBlocks(new Set([firstChanged.blockId]));
    }
  };

  const computeWordDiff = (original: string, current: string): WordDiff[] => {
    if (original === current) {
      return [{ type: 'unchanged', text: original }];
    }

    // Use diff-match-patch for character-level diff
    const dmp = new Diff.diff_match_patch();
    const diffs = dmp.diff_main(original, current);
    dmp.diff_cleanupSemantic(diffs);

    // Convert to word diffs
    const wordDiffs: WordDiff[] = [];
    
    diffs.forEach(([operation, text]) => {
      if (operation === 0) {
        wordDiffs.push({ type: 'unchanged', text });
      } else if (operation === 1) {
        wordDiffs.push({ type: 'added', text });
      } else if (operation === -1) {
        wordDiffs.push({ type: 'removed', text });
      }
    });

    return wordDiffs;
  };

  const toggleBlock = (blockId: string) => {
    setExpandedBlocks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(blockId)) {
        newSet.delete(blockId);
      } else {
        newSet.add(blockId);
      }
      return newSet;
    });
  };

  const changedBlocks = blockDiffs.filter(b => b.hasChanges);
  const unchangedBlocks = blockDiffs.filter(b => !b.hasChanges);

  const totalLines = blockDiffs.reduce((sum, b) => sum + b.original.split('\n').length, 0);
  const changedLines = changedBlocks.reduce((sum, b) => {
    const origLines = b.original.split('\n').length;
    const currLines = b.current.split('\n').length;
    return sum + Math.max(origLines, currLines);
  }, 0);

  const navigateToChange = (direction: 'prev' | 'next') => {
    if (changedBlocks.length === 0) return;
    
    let newIndex = currentChangeIndex;
    if (direction === 'next') {
      newIndex = (currentChangeIndex + 1) % changedBlocks.length;
    } else {
      newIndex = currentChangeIndex - 1;
      if (newIndex < 0) newIndex = changedBlocks.length - 1;
    }
    
    setCurrentChangeIndex(newIndex);
    const targetBlock = changedBlocks[newIndex];
    
    // Expand the block and scroll to it
    setExpandedBlocks(prev => new Set([...prev, targetBlock.blockId]));
    
    // Scroll to block
    setTimeout(() => {
      const element = document.getElementById(`diff-block-${targetBlock.blockId}`);
      element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  };

  const renderWordDiffs = (wordDiffs: WordDiff[]) => {
    return (
      <div className="whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">
        {wordDiffs.map((diff, idx) => {
          if (diff.type === 'unchanged') {
            return <span key={idx} className="text-neutral-700">{diff.text}</span>;
          } else if (diff.type === 'added') {
            return (
              <span key={idx} className="bg-green-200 text-green-900 px-0.5 rounded">
                {diff.text}
              </span>
            );
          } else {
            return (
              <span key={idx} className="bg-red-200 text-red-900 line-through px-0.5 rounded">
                {diff.text}
              </span>
            );
          }
        })}
      </div>
    );
  };

  return (
    <div className="h-full overflow-y-auto bg-neutral-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Summary Card */}
        <div className="mb-6 bg-white rounded-lg shadow-sm border border-neutral-200 p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <FileText className="w-6 h-6 text-blue-600" />
              <div>
                <h2 className="text-xl font-semibold text-neutral-900">
                  Document Comparison
                </h2>
                <p className="text-sm text-neutral-600 mt-1">
                  Side-by-side view: Original vs Current Edit
                </p>
              </div>
            </div>

            {/* Navigation Buttons */}
            {changedBlocks.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-neutral-600 mr-2">
                  Change {currentChangeIndex + 1} of {changedBlocks.length}
                </span>
                <button
                  onClick={() => navigateToChange('prev')}
                  className="p-2 hover:bg-neutral-100 rounded transition-colors"
                  title="Previous change"
                >
                  <ChevronUp className="w-4 h-4" />
                </button>
                <button
                  onClick={() => navigateToChange('next')}
                  className="p-2 hover:bg-neutral-100 rounded transition-colors"
                  title="Next change"
                >
                  <ChevronDown className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          {/* Statistics */}
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span className="text-neutral-700">
                <strong>{changedBlocks.length}</strong> block{changedBlocks.length !== 1 ? 's' : ''} changed
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span className="text-neutral-700">
                <strong>{wordDiffs => wordDiffs.filter(d => d.type === 'added').length}</strong> additions
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span className="text-neutral-700">
                <strong>{wordDiffs => wordDiffs.filter(d => d.type === 'removed').length}</strong> deletions
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-neutral-300 rounded"></div>
              <span className="text-neutral-700">
                <strong>{unchangedBlocks.length}</strong> unchanged
              </span>
            </div>
          </div>
        </div>

        {/* Changed Blocks */}
        {changedBlocks.length > 0 && (
          <div className="mb-6 space-y-3">
            {changedBlocks.map((blockDiff) => {
              const isExpanded = expandedBlocks.has(blockDiff.blockId);
              
              return (
                <div
                  key={blockDiff.blockId}
                  id={`diff-block-${blockDiff.blockId}`}
                  className="bg-white rounded-lg shadow-sm border-2 border-blue-200 overflow-hidden transition-all"
                >
                  {/* Block Header - Clickable */}
                  <button
                    onClick={() => toggleBlock(blockDiff.blockId)}
                    className="w-full bg-blue-50 px-4 py-3 border-b border-blue-200 hover:bg-blue-100 transition-colors text-left"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-blue-600" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-blue-600" />
                        )}
                        <span className="text-sm font-semibold text-blue-900">
                          Block #{blockDiff.blockNum}
                        </span>
                        <span className="text-xs text-blue-700 px-2 py-0.5 bg-blue-200 rounded">
                          {blockDiff.blockType}
                        </span>
                        <span className="text-xs text-blue-700">
                          {isExpanded ? 'Click to collapse' : 'Click to expand'}
                        </span>
                      </div>
                      {blockDiff.reason && (
                        <div className="flex items-center gap-2 text-xs text-blue-700">
                          <AlertCircle className="w-3 h-3" />
                          <span className="max-w-md truncate">{blockDiff.reason}</span>
                        </div>
                      )}
                    </div>
                  </button>

                  {/* Diff Content - Side by Side */}
                  {isExpanded && (
                    <div className="grid grid-cols-2 gap-0 divide-x divide-neutral-200">
                      {/* Original - Left */}
                      <div className="p-4 bg-red-50/30">
                        <div className="text-xs font-semibold text-red-800 mb-2 flex items-center gap-2">
                          <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                          ORIGINAL
                        </div>
                        <div className="text-sm text-neutral-700 font-mono whitespace-pre-wrap break-words leading-relaxed">
                          {blockDiff.original || <span className="italic text-neutral-400">(empty)</span>}
                        </div>
                      </div>

                      {/* Current - Right with word-level highlighting */}
                      <div className="p-4 bg-green-50/30">
                        <div className="text-xs font-semibold text-green-800 mb-2 flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          CURRENT
                        </div>
                        {renderWordDiffs(blockDiff.wordDiffs)}
                      </div>
                    </div>
                  )}

                  {/* Collapsed Preview */}
                  {!isExpanded && (
                    <div className="px-4 py-2 text-xs text-neutral-600 bg-neutral-50">
                      <span className="font-medium">Preview:</span> {blockDiff.current.substring(0, 100)}
                      {blockDiff.current.length > 100 && '...'}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Unchanged Blocks - Collapsed by default */}
        {unchangedBlocks.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-neutral-200 p-4">
            <div className="flex items-center gap-2 mb-2">
              <ChevronRight className="w-4 h-4 text-neutral-400" />
              <h3 className="text-sm font-semibold text-neutral-700">
                {unchangedBlocks.length} Unchanged Block{unchangedBlocks.length !== 1 ? 's' : ''}
              </h3>
            </div>
            <p className="text-xs text-neutral-600 ml-6">
              Blocks: {unchangedBlocks.map(b => `#${b.blockNum}`).join(', ')}
            </p>
          </div>
        )}

        {/* No changes message */}
        {changedBlocks.length === 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-neutral-200 p-12 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
              <AlertCircle className="w-8 h-8 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-neutral-900 mb-2">No Changes Detected</h3>
            <p className="text-neutral-600 text-sm">
              The document content matches the original version.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

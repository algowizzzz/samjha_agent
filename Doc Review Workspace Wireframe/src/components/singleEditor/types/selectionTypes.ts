/**
 * Selection data types for SingleDocumentEditor
 */

export interface SelectionData {
  selectedText: string;
  blockIds: string[];
  selectionScope: 'text' | 'blocks';
  currentBlockType: string;
  currentBlockLevel?: number;
  currentListStyle?: 'disc' | 'decimal';
  isConvertible: boolean;
}


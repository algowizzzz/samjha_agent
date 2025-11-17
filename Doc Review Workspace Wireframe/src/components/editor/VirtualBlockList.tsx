import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import type { Block } from './types';

interface VirtualBlockListProps {
  blocks: Block[];
  itemHeight?: number;
  renderBlock: (block: Block, index: number) => React.ReactNode;
  enabled?: boolean;
}

export function VirtualBlockList({ 
  blocks, 
  itemHeight = 80, 
  renderBlock, 
  enabled = true 
}: VirtualBlockListProps) {
  // If disabled or small number of blocks, render normally
  if (!enabled || blocks.length < 50) {
    return (
      <div>
        {blocks.map((block, index) => (
          <div key={block.id}>
            {renderBlock(block, index)}
          </div>
        ))}
      </div>
    );
  }

  // Virtual scrolling for large lists
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const block = blocks[index];
    return (
      <div style={style}>
        {renderBlock(block, index)}
      </div>
    );
  };

  return (
    <AutoSizer>
      {({ height, width }) => (
        <List
          height={height}
          itemCount={blocks.length}
          itemSize={itemHeight}
          width={width}
          className="virtual-block-list"
        >
          {Row}
        </List>
      )}
    </AutoSizer>
  );
}


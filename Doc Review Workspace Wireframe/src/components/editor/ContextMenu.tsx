import {
  Copy, Clipboard, Trash2, ArrowUp, ArrowDown,
  Type, MessageSquare, Sparkles, MoreHorizontal
} from 'lucide-react';

interface ContextMenuProps {
  position: { x: number; y: number };
  onClose: () => void;
  onCopy: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onTurnInto: () => void;
  onComment: () => void;
  onAskAI: () => void;
}

export function ContextMenu({
  position,
  onClose,
  onCopy,
  onDuplicate,
  onDelete,
  onMoveUp,
  onMoveDown,
  onTurnInto,
  onComment,
  onAskAI,
}: ContextMenuProps) {
  const menuItems = [
    { icon: <Copy className="w-4 h-4" />, label: 'Copy', onClick: onCopy },
    { icon: <Clipboard className="w-4 h-4" />, label: 'Duplicate', onClick: onDuplicate, shortcut: '⌘D' },
    { divider: true },
    { icon: <ArrowUp className="w-4 h-4" />, label: 'Move Up', onClick: onMoveUp, shortcut: '⌘↑' },
    { icon: <ArrowDown className="w-4 h-4" />, label: 'Move Down', onClick: onMoveDown, shortcut: '⌘↓' },
    { divider: true },
    { icon: <Type className="w-4 h-4" />, label: 'Turn Into', onClick: onTurnInto },
    { icon: <MessageSquare className="w-4 h-4" />, label: 'Comment', onClick: onComment, shortcut: '⌘/' },
    { icon: <Sparkles className="w-4 h-4" />, label: 'Ask AI', onClick: onAskAI },
    { divider: true },
    { icon: <Trash2 className="w-4 h-4" />, label: 'Delete', onClick: onDelete, danger: true },
  ];

  return (
    <>
      {/* Backdrop to close menu */}
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
      />
      
      <div
        className="fixed z-50 bg-white border border-neutral-200 rounded-lg shadow-xl py-1 w-56 animate-in fade-in zoom-in-95 duration-100"
        style={{ left: position.x, top: position.y }}
      >
        {menuItems.map((item, index) => {
          if ('divider' in item) {
            return <div key={index} className="border-t border-neutral-200 my-1" />;
          }

          return (
            <button
              key={index}
              className={`w-full px-3 py-2 text-left text-sm flex items-center gap-3 transition-colors ${
                item.danger
                  ? 'hover:bg-red-50 text-red-600'
                  : 'hover:bg-neutral-50 text-neutral-900'
              }`}
              onClick={() => {
                item.onClick();
                onClose();
              }}
            >
              <span className={item.danger ? 'text-red-500' : 'text-neutral-500'}>
                {item.icon}
              </span>
              <span className="flex-1">{item.label}</span>
              {item.shortcut && (
                <span className="text-xs text-neutral-400">{item.shortcut}</span>
              )}
            </button>
          );
        })}
      </div>
    </>
  );
}


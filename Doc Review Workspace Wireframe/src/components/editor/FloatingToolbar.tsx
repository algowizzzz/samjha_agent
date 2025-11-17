interface FloatingToolbarProps {
  position: { x: number; y: number };
  onAI: () => void;
}

export function FloatingToolbar({ position, onAI }: FloatingToolbarProps) {
  return (
    <div
      className="fixed z-[99999] flex items-center bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-xl px-3 py-2 animate-in fade-in zoom-in-95 duration-100 font-semibold text-sm cursor-pointer transition-colors"
      style={{ 
        left: `${position.x}px`, 
        top: `${position.y}px`,
        transform: 'translateX(-50%)',
      }}
      onClick={onAI}
      title="Ask RiskGPT to improve selected text"
    >
      Ask RiskGPT
    </div>
  );
}


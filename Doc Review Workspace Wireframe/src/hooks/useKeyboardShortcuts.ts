import { useEffect } from 'react';

type KeyCombo = string; // e.g., "cmd+b", "ctrl+z", "shift+enter"

interface ShortcutHandler {
  [key: KeyCombo]: (event: KeyboardEvent) => void;
}

export function useKeyboardShortcuts(shortcuts: ShortcutHandler, enabled = true) {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdOrCtrl = isMac ? event.metaKey : event.ctrlKey;
      
      // Build key combo string
      const parts: string[] = [];
      if (cmdOrCtrl) parts.push(isMac ? 'cmd' : 'ctrl');
      if (event.shiftKey) parts.push('shift');
      if (event.altKey) parts.push('alt');
      
      const key = event.key.toLowerCase();
      parts.push(key);
      
      const combo = parts.join('+');
      
      // Check if we have a handler for this combo
      if (shortcuts[combo]) {
        event.preventDefault();
        shortcuts[combo](event);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled]);
}


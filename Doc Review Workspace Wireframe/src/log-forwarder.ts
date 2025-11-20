const isDev = import.meta.env.DEV;

if (isDev) {
  const originalLog = console.log;
  const originalError = console.error;
  const originalWarn = console.warn;

  function sendToServer(level: string, args: any[]) {
    try {
      const serializedArgs = args.map(arg => {
        if (typeof arg === 'object') {
          try {
            return JSON.stringify(arg);
          } catch {
            return String(arg);
          }
        }
        return String(arg);
      });

      fetch('/__logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, args: serializedArgs }),
        keepalive: true,
      }).catch(() => {}); // silently ignore failures
    } catch {
      // logging must not break the app
    }
  }

  console.log = (...args: any[]) => {
    originalLog(...args);
    sendToServer('log', args);
  };

  console.warn = (...args: any[]) => {
    originalWarn(...args);
    sendToServer('warn', args);
  };

  console.error = (...args: any[]) => {
    originalError(...args);
    sendToServer('error', args);
  };
}


import { useEffect, useState } from 'react';
import { activityLogger, type ActivityLog } from '@/utils/activityLogger';
import { ScrollArea } from './ui/scroll-area';

export function ActivityLogDisplay() {
  const [logs, setLogs] = useState<ActivityLog[]>([]);

  useEffect(() => {
    // Subscribe to activity logger
    const unsubscribe = activityLogger.subscribe((newLog) => {
      setLogs((prev) => [...prev, newLog]);
    });

    // Load existing logs
    setLogs(activityLogger.getLogs());

    return unsubscribe;
  }, []);

  const getLevelColor = (level: ActivityLog['level']) => {
    switch (level) {
      case 'success':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      case 'warning':
        return 'text-yellow-600';
      default:
        return 'text-neutral-600';
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit',
      hour12: false 
    });
  };

  return (
    <div className="border-t bg-white">
      <div className="flex items-center justify-between px-4 py-2 border-b">
        <h3 className="font-semibold text-sm">Activity Log</h3>
        <button
          onClick={() => activityLogger.clear()}
          className="text-xs text-neutral-500 hover:text-neutral-700"
        >
          Clear
        </button>
      </div>
      
      <ScrollArea className="h-40">
        <div className="p-2 space-y-1 font-mono text-xs">
          {logs.length === 0 ? (
            <div className="text-center text-neutral-400 py-4">
              No activity yet
            </div>
          ) : (
            logs.map((log) => (
              <div key={log.id} className="flex items-start gap-2 py-1">
                <span className="text-neutral-400 shrink-0">
                  {formatTime(log.timestamp)}
                </span>
                {log.icon && <span className="shrink-0">{log.icon}</span>}
                <span className={getLevelColor(log.level)}>
                  {log.message}
                </span>
                {log.details && (
                  <span className="text-neutral-400 text-xs">
                    {log.details}
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}


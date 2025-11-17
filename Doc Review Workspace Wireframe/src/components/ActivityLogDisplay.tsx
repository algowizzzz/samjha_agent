import { useEffect, useState } from 'react';
import { activityLogger, type ActivityLog } from '@/utils/activityLogger';
import { ScrollArea } from './ui/scroll-area';

interface ActivityLogDisplayProps {
  backendLogs?: Array<string | { node?: string; msg?: string; timestamp?: string }>;
  backendErrors?: string[];
}

export function ActivityLogDisplay({ backendLogs = [], backendErrors = [] }: ActivityLogDisplayProps) {
  const [frontendLogs, setFrontendLogs] = useState<ActivityLog[]>([]);
  const [updateTrigger, setUpdateTrigger] = useState(0);

  useEffect(() => {
    // Subscribe to frontend activity logger
    const unsubscribe = activityLogger.subscribe((newLog) => {
      setFrontendLogs(activityLogger.getLogs()); // Always get fresh logs
      setUpdateTrigger(prev => prev + 1);
    });

    // Load existing frontend logs
    setFrontendLogs(activityLogger.getLogs());

    return unsubscribe;
  }, []);

  // Refresh frontend logs when clear is called
  const handleClear = () => {
    activityLogger.clear();
    setFrontendLogs([]);
    setUpdateTrigger(prev => prev + 1);
  };

  // Merge and format all logs
  const allLogs = [
    ...frontendLogs.map(log => ({
      timestamp: new Date(log.timestamp),
      type: 'frontend' as const,
      level: log.level,
      icon: log.icon,
      message: log.message,
      details: log.details,
    })),
    ...backendErrors.map(error => ({
      timestamp: new Date(),
      type: 'backend' as const,
      level: 'error' as const,
      icon: '⚠️',
      message: error,
    })),
    ...backendLogs.map(log => {
      const logText = typeof log === 'string' ? log : (log.msg || JSON.stringify(log));
      return {
        timestamp: typeof log === 'object' && log.timestamp ? new Date(log.timestamp) : new Date(),
        type: 'backend' as const,
        level: 'info' as const,
        icon: '🔍',
        message: logText,
      };
    }),
  ].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()); // Most recent first

  const getLevelColor = (level: string) => {
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
      <div className="flex items-center justify-between px-4 py-2 border-b bg-neutral-50">
        <h3 className="font-semibold text-sm">
          All Activity ({allLogs.length} logs)
        </h3>
        <button
          onClick={handleClear}
          className="text-xs text-neutral-500 hover:text-neutral-700"
          title="Clear frontend UI action logs only"
        >
          Clear UI Logs
        </button>
      </div>
      
      <ScrollArea className="h-24">
        <div className="p-2 space-y-1 font-mono text-xs">
          {allLogs.length === 0 ? (
            <div className="text-center text-neutral-400 py-4">
              No activity yet
            </div>
          ) : (
            allLogs.map((log, idx) => (
              <div key={`${log.type}-${idx}`} className="flex items-start gap-2 py-1 px-2 hover:bg-neutral-50 rounded">
                <span className="text-neutral-400 shrink-0 w-16">
                  {formatTime(log.timestamp)}
                </span>
                {log.icon && <span className="shrink-0">{log.icon}</span>}
                <span className={getLevelColor(log.level)}>
                  {log.message}
                </span>
                {log.details && (
                  <span className="text-neutral-400 text-xs ml-2">
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


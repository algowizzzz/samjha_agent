/**
 * Activity Logger - Centralized logging for user actions and system events
 * Provides user-friendly messages for display in the UI log panel
 */

export type ActivityLogLevel = 'info' | 'success' | 'warning' | 'error';

export interface ActivityLog {
  id: string;
  timestamp: Date;
  level: ActivityLogLevel;
  message: string;
  icon?: string;
  details?: string;
}

type LogListener = (log: ActivityLog) => void;

class ActivityLoggerService {
  private logs: ActivityLog[] = [];
  private listeners: Set<LogListener> = new Set();
  private maxLogs = 100; // Keep last 100 logs

  /**
   * Subscribe to log events
   */
  subscribe(listener: LogListener): () => void {
    this.listeners.add(listener);
    // Return unsubscribe function
    return () => this.listeners.delete(listener);
  }

  /**
   * Get all current logs
   */
  getLogs(): ActivityLog[] {
    return [...this.logs];
  }

  /**
   * Clear all logs (only frontend logs, not backend logs)
   */
  clear(): void {
    this.logs = [];
    // Notify all listeners to refresh their state
    this.listeners.forEach(listener => {
      // Send a special clear notification (empty log array will be handled by subscribers)
    });
  }

  /**
   * Add a log entry
   */
  private addLog(level: ActivityLogLevel, message: string, icon?: string, details?: string): void {
    const log: ActivityLog = {
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      level,
      message,
      icon,
      details
    };

    this.logs.push(log);

    // Keep only last N logs
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    this.notifyListeners();
  }

  /**
   * Notify all listeners of log update
   */
  private notifyListeners(): void {
    const latestLog = this.logs[this.logs.length - 1];
    if (latestLog) {
      this.listeners.forEach(listener => listener(latestLog));
    }
  }

  // Public logging methods

  info(message: string, details?: string): void {
    this.addLog('info', message, '📋', details);
  }

  success(message: string, details?: string): void {
    this.addLog('success', message, '✅', details);
  }

  warning(message: string, details?: string): void {
    this.addLog('warning', message, '⚠️', details);
  }

  error(message: string, details?: string): void {
    this.addLog('error', message, '❌', details);
  }

  // Specific action loggers with friendly messages

  blockSelected(blockId: string): void {
    this.info(`Selected block ${blockId.substring(0, 8)}...`);
  }

  blockEdited(blockId: string): void {
    this.info(`Editing block ${blockId.substring(0, 8)}...`);
  }

  suggestionAccepted(blockId: string): void {
    this.success(`Applied suggestion to block ${blockId.substring(0, 8)}...`);
  }

  suggestionRejected(blockId: string): void {
    this.info(`Rejected suggestion for block ${blockId.substring(0, 8)}...`);
  }

  changesSaved(count?: number): void {
    const msg = count ? `Saved ${count} change${count > 1 ? 's' : ''}` : 'Changes saved';
    this.success(msg, '💾');
  }

  documentLoaded(fileId: string): void {
    this.success(`Loaded document: ${fileId}`);
  }

  phaseStarted(phase: string): void {
    this.info(`Starting ${phase}...`);
  }

  phaseCompleted(phase: string): void {
    this.success(`${phase} completed`);
  }

  apiRequest(endpoint: string): void {
    this.info(`Requesting: ${endpoint}`);
  }

  apiSuccess(endpoint: string): void {
    this.success(`Success: ${endpoint}`);
  }

  apiError(endpoint: string, error: string): void {
    this.error(`Failed: ${endpoint}`, error);
  }

  userAction(action: string, target?: string): void {
    const msg = target ? `${action}: ${target}` : action;
    this.info(msg);
  }
}

// Export singleton instance
export const activityLogger = new ActivityLoggerService();

// For debugging in console
if (typeof window !== 'undefined') {
  (window as any).activityLogger = activityLogger;
}


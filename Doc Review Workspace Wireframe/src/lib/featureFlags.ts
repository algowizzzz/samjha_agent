/**
 * Feature flag system for gradual rollout of new features.
 * Uses localStorage for persistence.
 */

export type FeatureFlag = 'singleEditor' | 'advancedTemplates' | 'aiComments';

interface FeatureFlagConfig {
  key: string;
  name: string;
  description: string;
  defaultEnabled: boolean;
}

const FLAGS: Record<FeatureFlag, FeatureFlagConfig> = {
  singleEditor: {
    key: 'feature:singleEditor',
    name: 'Single Lexical Editor',
    description: 'Use the new single-editor architecture instead of per-block editors',
    defaultEnabled: false,
  },
  advancedTemplates: {
    key: 'feature:advancedTemplates',
    name: 'Advanced Templates',
    description: 'Enable advanced template checking and enforcement',
    defaultEnabled: false,
  },
  aiComments: {
    key: 'feature:aiComments',
    name: 'AI-Powered Comments',
    description: 'Enable AI-powered comment suggestions and threading',
    defaultEnabled: false,
  },
};

/**
 * Check if a feature flag is enabled.
 * 
 * @param flag - Feature flag to check
 * @returns true if enabled, false otherwise
 */
export function isFeatureEnabled(flag: FeatureFlag): boolean {
  const config = FLAGS[flag];
  if (!config) return false;

  try {
    const stored = localStorage.getItem(config.key);
    if (stored !== null) {
      return stored === 'true';
    }
    return config.defaultEnabled;
  } catch {
    return config.defaultEnabled;
  }
}

/**
 * Enable a feature flag.
 * 
 * @param flag - Feature flag to enable
 */
export function enableFeature(flag: FeatureFlag): void {
  const config = FLAGS[flag];
  if (!config) return;

  try {
    localStorage.setItem(config.key, 'true');
    console.log(`[FeatureFlags] Enabled: ${config.name}`);
  } catch (e) {
    console.error(`[FeatureFlags] Failed to enable ${config.name}:`, e);
  }
}

/**
 * Disable a feature flag.
 * 
 * @param flag - Feature flag to disable
 */
export function disableFeature(flag: FeatureFlag): void {
  const config = FLAGS[flag];
  if (!config) return;

  try {
    localStorage.setItem(config.key, 'false');
    console.log(`[FeatureFlags] Disabled: ${config.name}`);
  } catch (e) {
    console.error(`[FeatureFlags] Failed to disable ${config.name}:`, e);
  }
}

/**
 * Toggle a feature flag.
 * 
 * @param flag - Feature flag to toggle
 * @returns New state (true if enabled, false if disabled)
 */
export function toggleFeature(flag: FeatureFlag): boolean {
  const currentState = isFeatureEnabled(flag);
  if (currentState) {
    disableFeature(flag);
  } else {
    enableFeature(flag);
  }
  return !currentState;
}

/**
 * Get all feature flags and their current state.
 * 
 * @returns Array of feature flags with their config and state
 */
export function getAllFeatureFlags(): Array<FeatureFlagConfig & { flag: FeatureFlag; enabled: boolean }> {
  return Object.entries(FLAGS).map(([flag, config]) => ({
    flag: flag as FeatureFlag,
    ...config,
    enabled: isFeatureEnabled(flag as FeatureFlag),
  }));
}

/**
 * Reset all feature flags to their default values.
 */
export function resetAllFeatureFlags(): void {
  Object.entries(FLAGS).forEach(([flag, config]) => {
    try {
      localStorage.removeItem(config.key);
    } catch (e) {
      console.error(`[FeatureFlags] Failed to reset ${config.name}:`, e);
    }
  });
  console.log('[FeatureFlags] All flags reset to defaults');
}

// Expose to window for easy debugging in console
if (typeof window !== 'undefined') {
  (window as any).featureFlags = {
    enable: enableFeature,
    disable: disableFeature,
    toggle: toggleFeature,
    isEnabled: isFeatureEnabled,
    getAll: getAllFeatureFlags,
    reset: resetAllFeatureFlags,
  };
}


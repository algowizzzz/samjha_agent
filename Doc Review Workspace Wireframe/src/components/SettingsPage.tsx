import { useState } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

export function SettingsPage() {
  const [defaultViewMode, setDefaultViewMode] = useState('editing');
  const [autosaveInterval, setAutosaveInterval] = useState('60');
  const [debugMode, setDebugMode] = useState(false);

  const handleSaveSettings = () => {
    console.log('Saving settings:', {
      defaultViewMode,
      autosaveInterval,
      debugMode,
    });
  };

  return (
    <div className="flex h-full bg-white">
      <div className="max-w-4xl mx-auto w-full px-8 py-8">
        <h1 className="text-neutral-900 mb-8">Settings</h1>

        <div className="space-y-8">
          {/* General Settings */}
          <section>
            <h2 className="text-neutral-900 mb-4">General Settings</h2>
            <div className="space-y-4 bg-neutral-50 p-6 rounded-lg border border-neutral-200">
              <div>
                <label className="block text-neutral-700 mb-2">
                  Default Viewing Mode
                </label>
                <select
                  value={defaultViewMode}
                  onChange={(e) => setDefaultViewMode(e.target.value)}
                  className="w-full px-3 py-2 border border-neutral-300 rounded bg-white text-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-900"
                >
                  <option value="editing">Editing</option>
                  <option value="diff">Diff</option>
                </select>
              </div>

              <div>
                <label className="block text-neutral-700 mb-2">
                  Autosave Interval (seconds)
                </label>
                <input
                  type="number"
                  value={autosaveInterval}
                  onChange={(e) => setAutosaveInterval(e.target.value)}
                  min="10"
                  max="300"
                  className="w-full px-3 py-2 border border-neutral-300 rounded bg-white text-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-900"
                />
              </div>
            </div>
          </section>

          {/* Advanced Settings */}
          <section>
            <h2 className="text-neutral-900 mb-4">Advanced</h2>
            <div className="space-y-4 bg-neutral-50 p-6 rounded-lg border border-neutral-200">
              <div>
                <label className="block text-neutral-700 mb-2">
                  API Key Management
                </label>
                <div className="flex gap-2">
                  <input
                    type="password"
                    placeholder="••••••••••••••••"
                    className="flex-1 px-3 py-2 border border-neutral-300 rounded bg-white text-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-900"
                  />
                  <Button variant="outline">Update</Button>
                </div>
                <p className="text-neutral-600 mt-1">
                  API key for external integrations
                </p>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-neutral-700">Debug Views</label>
                  <p className="text-neutral-600">
                    Show debug information in activity logs
                  </p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={debugMode}
                    onChange={(e) => setDebugMode(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-neutral-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-neutral-900 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-neutral-900"></div>
                </label>
              </div>
            </div>
          </section>

          {/* Admin Controls */}
          <section>
            <h2 className="text-neutral-900 mb-4">Admin Controls</h2>
            <div className="space-y-4 bg-neutral-50 p-6 rounded-lg border border-neutral-200">
              <div>
                <label className="block text-neutral-700 mb-3">
                  Roles & Permissions
                </label>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-3 bg-white border border-neutral-200 rounded">
                    <div>
                      <p className="text-neutral-900">Admin</p>
                      <p className="text-neutral-600">Full system access</p>
                    </div>
                    <Badge variant="default">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-white border border-neutral-200 rounded">
                    <div>
                      <p className="text-neutral-900">Reviewer</p>
                      <p className="text-neutral-600">Can review and comment</p>
                    </div>
                    <Badge variant="default">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-white border border-neutral-200 rounded">
                    <div>
                      <p className="text-neutral-900">Viewer</p>
                      <p className="text-neutral-600">Read-only access</p>
                    </div>
                    <Badge variant="outline">Inactive</Badge>
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-neutral-700 mb-2">
                  User Management
                </label>
                <div className="flex gap-2">
                  <input
                    type="email"
                    placeholder="user@example.com"
                    className="flex-1 px-3 py-2 border border-neutral-300 rounded bg-white text-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-900"
                  />
                  <Button variant="outline">Add User</Button>
                </div>
              </div>
            </div>
          </section>

          {/* Save Button */}
          <div className="flex justify-end pt-4">
            <Button onClick={handleSaveSettings}>
              Save Settings
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

import { FileText, LayoutTemplate, Settings, MessageSquare } from 'lucide-react';

type Page = 'documents' | 'workspace' | 'templates' | 'prompts' | 'settings';

interface MainNavProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

export function MainNav({ currentPage, onNavigate }: MainNavProps) {
  return (
    <nav className="h-14 border-b border-neutral-200 bg-white px-6 flex items-center justify-between flex-shrink-0">
      <div className="flex items-center gap-8">
        <h1 className="text-neutral-900">Doc Review</h1>
        
        <div className="flex gap-1">
          <button
            onClick={() => onNavigate('documents')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded transition-colors ${
              currentPage === 'documents' || currentPage === 'workspace'
                ? 'bg-neutral-100 text-neutral-900'
                : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Documents</span>
          </button>
          
          <button
            onClick={() => onNavigate('templates')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded transition-colors ${
              currentPage === 'templates'
                ? 'bg-neutral-100 text-neutral-900'
                : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
            }`}
          >
            <LayoutTemplate className="w-4 h-4" />
            <span>Templates</span>
          </button>
          
          <button
            onClick={() => onNavigate('prompts')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded transition-colors ${
              currentPage === 'prompts'
                ? 'bg-neutral-100 text-neutral-900'
                : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Prompts</span>
          </button>
          
          <button
            onClick={() => onNavigate('settings')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded transition-colors ${
              currentPage === 'settings'
                ? 'bg-neutral-100 text-neutral-900'
                : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
            }`}
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>
    </nav>
  );
}

import { useState } from 'react';
import { PhaseFailed, NoTemplates, NoArtifacts, NoDocuments, NoResults } from './ErrorStates';
import { Button } from './ui/button';

type DemoState = 'phase-failed' | 'no-templates' | 'no-artifacts' | 'no-documents' | 'no-results';

export function ErrorStatesDemo() {
  const [currentState, setCurrentState] = useState<DemoState>('phase-failed');

  return (
    <div className="h-screen flex flex-col bg-neutral-50">
      {/* State Selector */}
      <div className="bg-white border-b border-neutral-200 p-4">
        <h2 className="text-neutral-900 mb-3">Error State Examples</h2>
        <div className="flex gap-2 flex-wrap">
          <Button
            variant={currentState === 'phase-failed' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCurrentState('phase-failed')}
          >
            Phase Failed
          </Button>
          <Button
            variant={currentState === 'no-templates' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCurrentState('no-templates')}
          >
            No Templates
          </Button>
          <Button
            variant={currentState === 'no-artifacts' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCurrentState('no-artifacts')}
          >
            No Artifacts
          </Button>
          <Button
            variant={currentState === 'no-documents' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCurrentState('no-documents')}
          >
            No Documents
          </Button>
          <Button
            variant={currentState === 'no-results' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCurrentState('no-results')}
          >
            No Results
          </Button>
        </div>
      </div>

      {/* State Display */}
      <div className="flex-1">
        {currentState === 'phase-failed' && (
          <PhaseFailed
            phaseName="Phase 2"
            errorMessage="Phase 2 failed due to validation error. The document structure could not be parsed correctly."
            onRetry={() => console.log('Retrying...')}
            onViewLogs={() => console.log('Viewing logs...')}
          />
        )}
        
        {currentState === 'no-templates' && (
          <NoTemplates onAddTemplate={() => console.log('Adding template...')} />
        )}
        
        {currentState === 'no-artifacts' && <NoArtifacts />}
        
        {currentState === 'no-documents' && <NoDocuments />}
        
        {currentState === 'no-results' && (
          <NoResults message="No documents match your search criteria" />
        )}
      </div>
    </div>
  );
}

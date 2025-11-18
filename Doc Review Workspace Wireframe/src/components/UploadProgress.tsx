import { Check, Loader2 } from 'lucide-react';

interface UploadProgressProps {
  steps: Array<{
    name: string;
    status: 'pending' | 'in_progress' | 'completed' | 'error';
    message?: string;
  }>;
}

export function UploadProgress({ steps }: UploadProgressProps) {
  return (
    <div className="space-y-3">
      {steps.map((step, index) => (
        <div key={index} className="flex items-center gap-3">
          {/* Icon */}
          <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
            step.status === 'completed' 
              ? 'bg-green-100' 
              : step.status === 'in_progress' 
              ? 'bg-blue-100' 
              : step.status === 'error'
              ? 'bg-red-100'
              : 'bg-neutral-100'
          }`}>
            {step.status === 'completed' ? (
              <Check className="w-4 h-4 text-green-600" />
            ) : step.status === 'in_progress' ? (
              <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
            ) : step.status === 'error' ? (
              <span className="text-red-600 text-xs">✗</span>
            ) : (
              <span className="text-neutral-400 text-xs">{index + 1}</span>
            )}
          </div>
          
          {/* Text */}
          <div className="flex-1">
            <p className={`text-sm font-medium ${
              step.status === 'completed' 
                ? 'text-green-700' 
                : step.status === 'in_progress' 
                ? 'text-blue-700' 
                : step.status === 'error'
                ? 'text-red-700'
                : 'text-neutral-500'
            }`}>
              {step.name}
            </p>
            {step.message && (
              <p className="text-xs text-neutral-500 mt-0.5">{step.message}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}


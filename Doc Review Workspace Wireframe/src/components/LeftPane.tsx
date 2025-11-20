import React, { useEffect, useState } from 'react';
import { ChevronRight, ChevronDown, FileText, AlertCircle, CheckCircle2, XCircle, AlertTriangle, Play } from 'lucide-react';
import { getDocument } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface LeftPaneProps {
  fileId?: string;
  onIssueSelect?: (id: string | null) => void;
  selectedIssueId?: string | null;
  onArtifactSelect?: (artifact: any) => void;
  suggestions?: Array<any>;
  onSuggestionSelect?: (id: string) => void;
  selectedSuggestionId?: string | null;
  onAcceptSuggestion?: (id: string) => void;
  onRejectSuggestion?: (id: string) => void;
  onCommentSuggestion?: (id: string) => void;
  onAnalyzeDocument?: () => void;
  isAnalyzing?: boolean;
}

interface AnalysisSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  analysis?: string;
  suggestions?: any[];
}

export function LeftPane({
  fileId,
  onAnalyzeDocument,
  isAnalyzing = false,
}: LeftPaneProps) {
  const [sections, setSections] = useState<AnalysisSection[]>([]);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [expandedSubsections, setExpandedSubsections] = useState<Record<string, { analysis: boolean; suggestions: boolean }>>({});

  useEffect(() => {
    async function loadAnalysisData() {
      if (!fileId) {
        setSections([]);
        return;
      }

      try {
        const response = await getDocument(fileId);
        const doc = response?.document || response; // Handle both {document: ...} and direct {...}
        const state = doc?.state || {};
        const phase1 = state.phase1 || {};
        const phase2_data = state.phase2_data || {};

        const analysisData: AnalysisSection[] = [
          {
            id: 'toc_review',
            title: '1. TOC & Structure Review',
            icon: <FileText className="w-4 h-4" />,
            analysis: phase1.toc_review,
            suggestions: [],
          },
          {
            id: 'conceptual_coverage',
            title: '2. Conceptual Coverage',
            icon: <FileText className="w-4 h-4" />,
            analysis: phase2_data.conceptual_coverage,
            suggestions: [],
          },
          {
            id: 'compliance_governance',
            title: '3. Compliance & Governance',
            icon: <FileText className="w-4 h-4" />,
            analysis: phase2_data.compliance_governance,
            suggestions: [],
          },
          {
            id: 'language_clarity',
            title: '4. Language & Clarity',
            icon: <FileText className="w-4 h-4" />,
            analysis: phase2_data.language_clarity,
            suggestions: [],
          },
          {
            id: 'structural_presentation',
            title: '5. Structural & Presentation',
            icon: <FileText className="w-4 h-4" />,
            analysis: phase2_data.structural_presentation,
            suggestions: [],
          },
          {
            id: 'synthesis',
            title: '6. 📊 Synthesis Summary',
            icon: <FileText className="w-4 h-4" />,
            analysis: phase2_data.synthesis,
            suggestions: [],
          },
        ];

        setSections(analysisData);

        // Initialize all subsections as collapsed
        const initialSubsections: Record<string, { analysis: boolean; suggestions: boolean }> = {};
        analysisData.forEach(section => {
          initialSubsections[section.id] = { analysis: false, suggestions: false };
        });
        setExpandedSubsections(initialSubsections);
      } catch (error) {
        console.error('[LeftPane] Failed to load analysis:', error);
        setSections([]);
      }
    }

    loadAnalysisData();
  }, [fileId]);

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  const toggleSubsection = (sectionId: string, subsection: 'analysis' | 'suggestions') => {
    setExpandedSubsections(prev => ({
      ...prev,
      [sectionId]: {
        ...prev[sectionId],
        [subsection]: !prev[sectionId]?.[subsection],
      },
    }));
  };

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="flex-shrink-0 px-5 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h2 className="text-sm font-semibold text-gray-900">
              Document Analysis
            </h2>
            <p className="mt-1 text-xs text-gray-500">
              Review AI-generated insights and suggestions
            </p>
          </div>
          {onAnalyzeDocument && (
            <button
              onClick={onAnalyzeDocument}
              disabled={isAnalyzing || !fileId}
              className="ml-3 inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? (
                <>
                  <span className="animate-spin">⏳</span>
                  Analyzing...
                </>
              ) : (
                <>
                  <Play className="w-3 h-3" />
                  Analyze
                </>
              )}
            </button>
          )}
        </div>
        {sections.some(s => s.analysis) && (
          <div className="mt-3 inline-flex items-center rounded-full border border-green-200 bg-green-50 px-3 py-1 text-xs font-medium text-green-700">
            {sections.filter(s => s.analysis).length} sections analyzed
          </div>
        )}
      </div>

      {/* Analysis Sections */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {!fileId && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <AlertCircle className="w-10 h-10 text-gray-300 mb-3" />
            <p className="text-sm text-gray-500">
              Select a document to view analysis
            </p>
          </div>
        )}

        {fileId && sections.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <AlertCircle className="w-10 h-10 text-gray-300 mb-3" />
            <p className="text-sm text-gray-500">
              Click "Analyze Document" to start
            </p>
          </div>
        )}

        {sections.map((section) => {
          const isExpanded = expandedSections.has(section.id);
          const hasAnalysis = !!section.analysis;
          const analysisExpanded = expandedSubsections[section.id]?.analysis || false;
          const suggestionsExpanded = expandedSubsections[section.id]?.suggestions || false;

          return (
            <div
              key={section.id}
              className="rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm hover:shadow-md transition-shadow"
            >
              {/* Section Header */}
              <button
                onClick={() => toggleSection(section.id)}
                className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {section.icon}
                  <span className="font-medium text-sm text-gray-900">{section.title}</span>
                </div>
                <div className="flex items-center gap-2">
                  {hasAnalysis && (
                    <span className="inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700">
                      Ready
                    </span>
                  )}
                  {isExpanded ? (
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  )}
                </div>
              </button>

              {/* Section Content */}
              {isExpanded && (
                <div className="border-t border-gray-100">
                  {/* Analysis Subsection */}
                  <div className="border-b border-gray-100">
                    <button
                      onClick={() => toggleSubsection(section.id, 'analysis')}
                      className="w-full flex items-center justify-between p-3 text-left hover:bg-gray-50 transition-colors"
                    >
                      <span className="text-xs font-semibold text-gray-700">Analysis</span>
                      {analysisExpanded ? (
                        <ChevronDown className="w-3 h-3 text-gray-400" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-gray-400" />
                      )}
                    </button>
                    {analysisExpanded && (
                      <div className="px-4 pb-4 max-h-96 overflow-y-auto bg-gradient-to-br from-white to-gray-50">
                        {hasAnalysis ? (
                          <div className="prose prose-sm max-w-none 
                            prose-headings:font-semibold prose-headings:text-gray-900 prose-headings:mb-3 prose-headings:mt-4
                            prose-h1:text-lg prose-h1:text-blue-900 prose-h1:border-b prose-h1:border-blue-100 prose-h1:pb-2
                            prose-h2:text-base prose-h2:text-blue-800
                            prose-h3:text-sm prose-h3:text-blue-700
                            prose-p:text-gray-700 prose-p:leading-relaxed prose-p:mb-3
                            prose-strong:text-gray-900 prose-strong:font-semibold prose-strong:text-blue-900
                            prose-em:text-gray-600 prose-em:italic
                            prose-ul:my-3 prose-ul:space-y-2
                            prose-ol:my-3 prose-ol:space-y-2
                            prose-li:text-gray-700 prose-li:leading-relaxed prose-li:marker:text-blue-600
                            prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline
                            prose-code:text-xs prose-code:bg-gray-100 prose-code:text-pink-600 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:font-mono prose-code:before:content-none prose-code:after:content-none
                            prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-pre:text-xs prose-pre:p-4 prose-pre:rounded-lg prose-pre:overflow-x-auto
                            prose-blockquote:border-l-4 prose-blockquote:border-blue-500 prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-gray-600 prose-blockquote:bg-blue-50 prose-blockquote:py-2
                            prose-table:text-xs prose-table:border-collapse
                            prose-th:bg-blue-100 prose-th:text-blue-900 prose-th:font-semibold prose-th:p-2 prose-th:border prose-th:border-blue-200
                            prose-td:p-2 prose-td:border prose-td:border-gray-200 prose-td:text-gray-700
                            prose-hr:border-gray-200 prose-hr:my-4">
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              components={{
                                h1: ({node, ...props}) => <h1 className="flex items-center gap-2" {...props}><CheckCircle2 className="w-4 h-4 text-blue-600 inline" />{props.children}</h1>,
                                h2: ({node, ...props}) => <h2 className="flex items-center gap-2" {...props}><FileText className="w-3.5 h-3.5 text-blue-600 inline" />{props.children}</h2>,
                                h3: ({node, ...props}) => <h3 className="flex items-center gap-2" {...props}><ChevronRight className="w-3 h-3 text-blue-600 inline" />{props.children}</h3>,
                                strong: ({node, ...props}) => {
                                  const text = String(props.children);
                                  if (text.toLowerCase().includes('gap') || text.toLowerCase().includes('issue') || text.toLowerCase().includes('missing')) {
                                    return <strong className="text-red-700 font-semibold" {...props}><AlertTriangle className="w-3 h-3 inline mr-1 text-red-600" />{props.children}</strong>;
                                  }
                                  if (text.toLowerCase().includes('strength') || text.toLowerCase().includes('complete') || text.toLowerCase().includes('excellent')) {
                                    return <strong className="text-green-700 font-semibold" {...props}><CheckCircle2 className="w-3 h-3 inline mr-1 text-green-600" />{props.children}</strong>;
                                  }
                                  return <strong className="text-blue-900 font-semibold" {...props} />;
                                },
                                ul: ({node, ...props}) => <ul className="space-y-2 ml-4" {...props} />,
                                ol: ({node, ...props}) => <ol className="space-y-2 ml-4" {...props} />,
                                li: ({node, ...props}) => (
                                  <li className="relative pl-2" {...props}>
                                    {props.children}
                                  </li>
                                ),
                                table: ({node, ...props}) => (
                                  <div className="my-4 overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
                                    <table className="min-w-full" {...props} />
                                  </div>
                                ),
                                blockquote: ({node, ...props}) => (
                                  <blockquote className="my-4 border-l-4 border-blue-500 bg-blue-50 p-4 rounded-r-lg" {...props}>
                                    <AlertCircle className="w-4 h-4 inline mr-2 text-blue-600" />
                                    {props.children}
                                  </blockquote>
                                ),
                              }}
                            >
                              {section.analysis}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-gray-400 py-4">
                            <AlertCircle className="w-4 h-4" />
                            <span className="text-xs">No analysis available yet</span>
                          </div>
            )}
          </div>
        )}
      </div>

                  {/* Suggestions Subsection */}
                  <div>
                    <button
                      onClick={() => toggleSubsection(section.id, 'suggestions')}
                      className="w-full flex items-center justify-between p-3 text-left hover:bg-gray-50 transition-colors"
                    >
                      <span className="text-xs font-semibold text-gray-700">Suggestions</span>
                      <div className="flex items-center gap-2">
                        <span className="inline-flex items-center rounded-full border border-gray-200 bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
                          Coming soon
                        </span>
                        {suggestionsExpanded ? (
                          <ChevronDown className="w-3 h-3 text-gray-400" />
                        ) : (
                          <ChevronRight className="w-3 h-3 text-gray-400" />
                        )}
                      </div>
                    </button>
                    {suggestionsExpanded && (
                      <div className="px-4 pb-4">
                        <div className="flex items-center gap-2 text-gray-400 py-4">
                          <AlertCircle className="w-4 h-4" />
                          <span className="text-xs">AI-generated suggestions will appear here</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

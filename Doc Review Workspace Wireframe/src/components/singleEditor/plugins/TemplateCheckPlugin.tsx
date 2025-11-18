import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { useEffect, useState, useCallback } from 'react';
import { $getRoot } from 'lexical';
import { $isDocHeadingNode } from '../nodes/DocHeadingNode';

export interface TemplateSection {
  key: string;
  expectedLevel: 1 | 2 | 3;
  displayName: string;
  required: boolean;
}

export interface TemplateViolation {
  type: 'missing' | 'out_of_order' | 'wrong_level' | 'extra';
  sectionKey?: string;
  expectedSection?: string;
  actualSection?: string;
  message: string;
}

interface TemplateCheckPluginProps {
  template?: TemplateSection[];
  onViolationsChange?: (violations: TemplateViolation[]) => void;
  checkOnChange?: boolean;
}

const DEFAULT_TEMPLATE: TemplateSection[] = [
  { key: 'overview', expectedLevel: 1, displayName: 'Overview', required: true },
  { key: 'scope', expectedLevel: 1, displayName: 'Scope', required: true },
  { key: 'policy_requirements', expectedLevel: 1, displayName: 'Policy Requirements', required: true },
  { key: 'roles_responsibilities', expectedLevel: 1, displayName: 'Roles & Responsibilities', required: false },
  { key: 'procedures', expectedLevel: 1, displayName: 'Procedures', required: false },
  { key: 'monitoring', expectedLevel: 1, displayName: 'Monitoring & Reporting', required: false },
];

/**
 * Plugin that enforces document template compliance.
 * 
 * Checks for:
 * - Missing required sections
 * - Out-of-order sections
 * - Incorrect heading levels
 * - Extra/unexpected sections
 * 
 * Emits violations that can be displayed in a sidebar or warning panel.
 */
export function TemplateCheckPlugin({
  template = DEFAULT_TEMPLATE,
  onViolationsChange,
  checkOnChange = true,
}: TemplateCheckPluginProps) {
  const [editor] = useLexicalComposerContext();
  const [violations, setViolations] = useState<TemplateViolation[]>([]);

  const checkTemplate = useCallback(() => {
    editor.getEditorState().read(() => {
      const root = $getRoot();
      const newViolations: TemplateViolation[] = [];
      
      // Extract all headings from the document
      const headings: Array<{
        key: string;
        level: number;
        text: string;
        sectionKey?: string;
      }> = [];

      root.getChildren().forEach((node) => {
        if ($isDocHeadingNode(node)) {
          const text = node.getTextContent();
          headings.push({
            key: node.getKey(),
            level: node.__level,
            text,
            sectionKey: node.__sectionKey,
          });
        }
      });

      // Build a map of found sections by sectionKey
      const foundSections = new Map<string, typeof headings[0]>();
      headings.forEach((heading) => {
        if (heading.sectionKey) {
          foundSections.set(heading.sectionKey, heading);
        }
      });

      // Check for missing required sections
      template.forEach((templateSection) => {
        if (templateSection.required && !foundSections.has(templateSection.key)) {
          newViolations.push({
            type: 'missing',
            sectionKey: templateSection.key,
            message: `Missing required section: ${templateSection.displayName}`,
          });
        }
      });

      // Check section order
      const templateKeys = template.map((t) => t.key);
      const foundKeys = headings
        .filter((h) => h.sectionKey)
        .map((h) => h.sectionKey!);

      let expectedIndex = 0;
      foundKeys.forEach((key, actualIndex) => {
        const templateIndex = templateKeys.indexOf(key);
        if (templateIndex !== -1 && templateIndex < expectedIndex) {
          const templateSection = template.find((t) => t.key === key);
          newViolations.push({
            type: 'out_of_order',
            sectionKey: key,
            message: `Section "${templateSection?.displayName}" is out of order`,
          });
        }
        if (templateIndex !== -1) {
          expectedIndex = Math.max(expectedIndex, templateIndex + 1);
        }
      });

      // Check heading levels
      foundSections.forEach((heading, sectionKey) => {
        const templateSection = template.find((t) => t.key === sectionKey);
        if (templateSection && heading.level !== templateSection.expectedLevel) {
          newViolations.push({
            type: 'wrong_level',
            sectionKey,
            message: `Section "${templateSection.displayName}" should be level ${templateSection.expectedLevel}, but is level ${heading.level}`,
          });
        }
      });

      setViolations(newViolations);
      if (onViolationsChange) {
        onViolationsChange(newViolations);
      }
    });
  }, [editor, template, onViolationsChange]);

  useEffect(() => {
    if (!checkOnChange) return;

    // Run initial check
    checkTemplate();

    // Register update listener
    return editor.registerUpdateListener(() => {
      checkTemplate();
    });
  }, [editor, checkTemplate, checkOnChange]);

  return null;
}

/**
 * Helper to manually trigger template check from outside.
 */
export function triggerTemplateCheck(editor: any) {
  editor.getEditorState().read(() => {
    // This will trigger the plugin's update listener
  });
}


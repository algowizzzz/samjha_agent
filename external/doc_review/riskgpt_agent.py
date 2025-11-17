"""
RiskGPT Agent - LangGraph-style orchestration for document review Q&A.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from external.doc_review.riskgpt_schemas import RiskGPTAgentState
from external.doc_review.riskgpt_nodes import (
    context_loader_node,
    intent_classifier_node,
    block_improver_node,
    chat_responder_node,
    doc_searcher_node,
    end_node,
)

logger = logging.getLogger(__name__)


class RiskGPTAgent:
    """
    Orchestrates the RiskGPT workflow using a state machine pattern.
    
    Flow:
    1. Context Loader → Load document, blocks, template
    2. Intent Classifier → Determine user intent
    3. One of:
       - Block Improver → Generate block suggestions
       - Chat Responder → Answer general questions
       - Doc Searcher → Search and answer about document
    4. End → Format final output
    """
    
    def __init__(self):
        self.max_steps = 10  # Prevent infinite loops
    
    def run(
        self,
        file_id: str,
        user_prompt: str,
        selected_block_ids: list,
        conversation_history: list,
        document_state: Dict[str, Any],
        template_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the RiskGPT workflow.
        
        Args:
            file_id: Document file ID
            user_prompt: User's question/request
            selected_block_ids: IDs of selected blocks (empty for general chat)
            conversation_history: Last 5-10 messages for context
            document_state: Full document state from store
            template_content: Template markdown (optional)
        
        Returns:
            Dict with:
                - analysis: Text response
                - suggestions: List of structured suggestions (if blocks selected)
                - logs: Processing logs
                - metrics: Timing metrics
        """
        # Initialize state
        state: RiskGPTAgentState = {
            "file_id": file_id,
            "user_prompt": user_prompt,
            "selected_block_ids": selected_block_ids or [],
            "conversation_history": conversation_history or [],
            "control": "context_loader",
            "logs": [],
            "metrics": {
                "start_time": datetime.utcnow().isoformat() + "Z",
                "node_timings": {}
            }
        }
        
        logger.info(f"[RiskGPTAgent] Starting workflow for file_id={file_id}, "
                   f"selected_blocks={len(selected_block_ids)}, prompt_len={len(user_prompt)}")
        
        try:
            step = 0
            while step < self.max_steps:
                step += 1
                control = state.get("control", "end")
                
                if control == "end":
                    break
                
                node_start = datetime.utcnow()
                
                # Route to appropriate node
                if control == "context_loader":
                    updates = context_loader_node(state, document_state, template_content)
                
                elif control == "intent_classifier":
                    updates = intent_classifier_node(state)
                
                elif control == "block_improver":
                    updates = block_improver_node(state)
                
                elif control == "chat_responder":
                    updates = chat_responder_node(state)
                
                elif control == "doc_searcher":
                    updates = doc_searcher_node(state)
                
                else:
                    logger.warning(f"[RiskGPTAgent] Unknown control signal: {control}")
                    break
                
                # Update state
                state.update(updates)
                
                # Track timing
                node_time = (datetime.utcnow() - node_start).total_seconds() * 1000
                state["metrics"]["node_timings"][control] = node_time
                
                logger.info(f"[RiskGPTAgent] Step {step}: {control} → {state.get('control')} "
                           f"({node_time:.1f}ms)")
            
            # Run end node to format output
            if state.get("control") != "end":
                logger.warning(f"[RiskGPTAgent] Reached max steps ({self.max_steps})")
            
            end_updates = end_node(state)
            state.update(end_updates)
            
            # Calculate total time
            start_time = datetime.fromisoformat(state["metrics"]["start_time"].replace("Z", ""))
            total_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["metrics"]["total_ms"] = total_ms
            state["metrics"]["steps"] = step
            
            logger.info(f"[RiskGPTAgent] Completed in {step} steps, {total_ms:.1f}ms total")
            
            # Return final output
            final_output = state.get("final_output", {
                "analysis": "No response generated",
                "suggestions": []
            })
            
            return {
                "analysis": final_output.get("analysis", ""),
                "suggestions": final_output.get("suggestions", []),
                "logs": state.get("logs", []),
                "metrics": state.get("metrics", {}),
                "intent": state.get("intent"),
                "intent_confidence": state.get("intent_confidence")
            }
            
        except Exception as e:
            logger.exception(f"[RiskGPTAgent] Error in workflow: {e}")
            return {
                "analysis": f"Error processing request: {str(e)}",
                "suggestions": [],
                "logs": state.get("logs", []),
                "metrics": state.get("metrics", {}),
                "error": str(e)
            }
    
    def _merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge updates into base state."""
        base.update(updates)
        return base


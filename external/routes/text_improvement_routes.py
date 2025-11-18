"""
Simple text improvement endpoint - character-based AI suggestions.
No block IDs, no file context, just quick text improvements.
"""
from flask import Blueprint, request, jsonify
import logging
import os
from anthropic import Anthropic

logger = logging.getLogger(__name__)

text_improvement_bp = Blueprint('text_improvement', __name__)


def get_llm_client():
    """Get Anthropic client for LLM calls."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=api_key)


@text_improvement_bp.route('/api/text-improvement/improve', methods=['POST'])
def improve_text():
    """
    Improve selected text with AI suggestions.
    
    Request:
        {
            "text": "selected text to improve",
            "context_before": "optional context",
            "context_after": "optional context",
            "instruction": "optional specific instruction"
        }
    
    Response:
        {
            "original": "input text",
            "improved": "improved text",
            "reason": "why it was improved"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        context_before = data.get('context_before', '')
        context_after = data.get('context_after', '')
        instruction = data.get('instruction', '')
        
        if not text.strip():
            return jsonify({"error": "Empty text provided"}), 400
        
        logger.info(f"[TextImprovement] Improving text: {text[:50]}...")
        
        # Call LLM
        client = get_llm_client()
        
        # Build prompt
        system_prompt = """You are a helpful documentation assistant. Your job is to improve text clarity, precision, and professionalism while preserving the original meaning.

Guidelines:
- Keep the same tone and style
- Fix grammar and clarity issues
- Make terminology more precise
- Keep it concise
- Preserve technical terms
- Only improve what needs improvement"""

        user_message = ""
        
        # Add context if provided
        if context_before:
            user_message += f"CONTEXT BEFORE:\n{context_before}\n\n"
        
        user_message += f"TEXT TO IMPROVE:\n{text}\n\n"
        
        if context_after:
            user_message += f"CONTEXT AFTER:\n{context_after}\n\n"
        
        # Add specific instruction if provided
        if instruction:
            user_message += f"SPECIFIC INSTRUCTION:\n{instruction}\n\n"
        
        user_message += """Please respond in JSON format:
{
  "improved": "your improved version of the text",
  "reason": "brief explanation of changes"
}

If the text is already good, return it unchanged with reason: "No improvements needed"."""

        # Call Claude (use model from env)
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )
        
        # Parse response
        response_text = response.content[0].text.strip()
        
        # Remove markdown code fences if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        import json
        result = json.loads(response_text)
        
        # Return structured response
        return jsonify({
            "original": text,
            "improved": result.get("improved", text),
            "reason": result.get("reason", "Improved by AI"),
            "success": True
        })
        
    except ValueError as e:
        logger.error(f"[TextImprovement] Configuration error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
    
    except Exception as e:
        logger.exception(f"[TextImprovement] Error: {e}")
        return jsonify({
            "error": f"Failed to improve text: {str(e)}",
            "success": False
        }), 500


@text_improvement_bp.route('/api/text-improvement/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if API key is configured
        api_key = os.getenv("ANTHROPIC_API_KEY")
        has_api_key = bool(api_key)
        
        return jsonify({
            "status": "healthy",
            "api_configured": has_api_key
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


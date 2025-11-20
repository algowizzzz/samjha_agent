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
        logger.info(f"[TextImprovement] LLM raw response: {response_text}")
        
        # Remove markdown code fences if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        import json
        import re
        
        # Try multiple strategies to extract JSON
        result = None
        
        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(response_text)
            logger.info(f"[TextImprovement] Direct JSON parse succeeded")
        except json.JSONDecodeError as e:
            logger.warning(f"[TextImprovement] Direct JSON parse failed: {e}")
            
            # Strategy 2: Try to find and extract just the JSON portion by brace matching
            try:
                # Find first { and last }
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = response_text[start:end+1]
                    result = json.loads(json_str)
                    logger.info(f"[TextImprovement] Successfully extracted JSON by brace matching")
            except (json.JSONDecodeError, ValueError) as e2:
                logger.warning(f"[TextImprovement] Brace matching failed: {e2}")
            
            # Strategy 3: Extract using more flexible regex (handles nested structures)
            if not result:
                try:
                    # Match JSON object that contains "improved" and "reason" keys
                    json_match = re.search(r'\{[\s\S]*?"improved"[\s\S]*?"reason"[\s\S]*?\}', response_text)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        logger.info(f"[TextImprovement] Successfully extracted JSON via regex")
                except (json.JSONDecodeError, ValueError) as e3:
                    logger.warning(f"[TextImprovement] Regex extraction failed: {e3}")
        
        # If all strategies failed, return error
        if not result:
            logger.error(f"[TextImprovement] All JSON parsing strategies failed")
            logger.error(f"[TextImprovement] Raw response: {response_text}")
            return jsonify({
                "error": "LLM returned invalid JSON format",
                "raw_response": response_text[:500],
                "success": False
            }), 500
        
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


"""
LLM Client for Agent
Provides unified interface for LLM interactions with Anthropic Claude.
"""
import os
import json
from typing import Dict, Any, Optional, List, Generator, Callable
from datetime import datetime

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  anthropic package not installed. Run: pip install anthropic")


class LLMClient:
    """Unified LLM client supporting Anthropic Claude"""
    
    def __init__(self):
        self.provider = None
        self.client = None
        self.model = None
        self.temperature = 0.2
        self.max_tokens = 4096
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM client based on available credentials"""
        # Try Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = Anthropic(api_key=api_key)
                self.provider = "anthropic"
                # Default to claude-3-opus (most capable, works with all API tiers)
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
                self.temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2"))
                self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
                print(f"✓ LLM initialized: Anthropic {self.model}")
                return
            except Exception as e:
                print(f"⚠️  Failed to initialize Anthropic: {e}")
        
        if self.demo_mode:
            print("✓ Demo mode enabled - using mock LLM responses")
        else:
            print("⚠️  No LLM configured. Set ANTHROPIC_API_KEY in .env or DEMO_MODE=true for demo")
    
    def is_available(self) -> bool:
        """Check if LLM is available (or demo mode is enabled)"""
        return self.client is not None or self.demo_mode
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None
    ) -> str:
        """
        Invoke LLM with messages.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: "json" to request JSON output
            
        Returns:
            LLM response text
        """
        if not self.is_available():
            raise RuntimeError("LLM not available")
        
        # Check for demo mode mock response
        if self.demo_mode and messages:
            user_prompt = messages[-1].get("content", "") if messages else ""
            mock_response = self._get_demo_mock_response(system or "", user_prompt, response_format)
            if mock_response:
                return mock_response
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == "anthropic":
            return self._invoke_anthropic(messages, system, temp, max_tok, response_format)
        
        raise RuntimeError(f"Unknown provider: {self.provider}")
    
    def invoke_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        response_format: Optional[str] = None
    ) -> str:
        """
        Simplified invoke with system and user prompts.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Override default temperature
            response_format: "json" to request JSON output
            
        Returns:
            LLM response text
        """
        # Check for demo mode mock response
        if self.demo_mode:
            mock_response = self._get_demo_mock_response(system_prompt, user_prompt, response_format)
            if mock_response:
                return mock_response
        
        messages = [{"role": "user", "content": user_prompt}]
        return self.invoke(messages, system=system_prompt, temperature=temperature, response_format=response_format)
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        """
        Stream LLM response token by token.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: "json" to request JSON output
            callback: Optional callback function for each chunk
            
        Yields:
            Text chunks as they arrive
        """
        if not self.is_available():
            raise RuntimeError("LLM not available")
        
        # Check for demo mode mock response
        if self.demo_mode and messages:
            user_prompt = messages[-1].get("content", "") if messages else ""
            mock_response = self._get_demo_mock_response(system or "", user_prompt, response_format)
            if mock_response:
                # Simulate streaming by yielding chunks
                chunk_size = 10
                for i in range(0, len(mock_response), chunk_size):
                    chunk = mock_response[i:i + chunk_size]
                    if callback:
                        callback(chunk)
                    yield chunk
                return
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == "anthropic":
            for chunk in self._stream_anthropic(messages, system, temp, max_tok, response_format, callback):
                yield chunk
        else:
            raise RuntimeError(f"Unknown provider: {self.provider}")
    
    def stream_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        response_format: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        """
        Simplified stream with system and user prompts.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Override default temperature
            response_format: "json" to request JSON output
            callback: Optional callback function for each chunk
            
        Yields:
            Text chunks as they arrive
        """
        # Check for demo mode mock response
        if self.demo_mode:
            mock_response = self._get_demo_mock_response(system_prompt, user_prompt, response_format)
            if mock_response:
                # Simulate streaming by yielding chunks
                chunk_size = 10
                for i in range(0, len(mock_response), chunk_size):
                    chunk = mock_response[i:i + chunk_size]
                    if callback:
                        callback(chunk)
                    yield chunk
                return
        
        messages = [{"role": "user", "content": user_prompt}]
        for chunk in self.stream(messages, system=system_prompt, temperature=temperature, response_format=response_format, callback=callback):
            yield chunk
    
    def _invoke_anthropic(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: Optional[str]
    ) -> str:
        """Invoke Anthropic Claude API"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system:
            kwargs["system"] = system
        
        # For JSON responses, add instruction to system prompt
        if response_format == "json":
            json_instruction = "\n\nIMPORTANT: Respond with valid JSON only, no markdown formatting or code blocks."
            if system:
                kwargs["system"] = system + json_instruction
            else:
                kwargs["system"] = json_instruction
        
        try:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def _stream_anthropic(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: Optional[str],
        callback: Optional[Callable[[str], None]]
    ) -> Generator[str, None, None]:
        """Stream Anthropic Claude API response"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system:
            kwargs["system"] = system
        
        # For JSON responses, add instruction to system prompt
        if response_format == "json":
            json_instruction = "\n\nIMPORTANT: Respond with valid JSON only, no markdown formatting or code blocks."
            if system:
                kwargs["system"] = system + json_instruction
            else:
                kwargs["system"] = json_instruction
        
        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text_block in stream.text_stream:
                    if text_block:
                        if callback:
                            callback(text_block)
                        yield text_block
        except Exception as e:
            raise RuntimeError(f"Anthropic API streaming error: {e}")
    
    def _get_demo_mock_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None
    ) -> Optional[str]:
        """
        Get mock response for demo mode.
        Returns mock data for specific queries, especially "Show me limits with high utilization"
        """
        # Check if this is the high utilization query
        user_prompt_lower = user_prompt.lower()
        if "high utilization" in user_prompt_lower or "utilization" in user_prompt_lower:
            if response_format == "json":
                # Mock SQL generation response
                return json.dumps({
                    "sql": "SELECT * FROM limits_data WHERE utilization > 0.95 AND date = (SELECT MAX(date) FROM limits_data) ORDER BY utilization DESC LIMIT 20",
                    "reasoning": "Query requests limits with high utilization. Filtering for utilization > 0.95 (95%) to show limits that are near or at capacity. Using latest date snapshot. Ordering by utilization descending to show highest first.",
                    "plan_quality": "high",
                    "confidence": 0.95
                })
            else:
                # Mock final response synthesis - try to extract actual row count from prompt
                row_count = 4  # Default
                try:
                    # Try to extract row_count from the user_prompt (which contains state_summary)
                    import re
                    # Look for "row_count": number in the prompt
                    match = re.search(r'"row_count":\s*(\d+)', user_prompt)
                    if match:
                        row_count = int(match.group(1))
                except:
                    pass
                
                # Generate response that matches actual row count
                if row_count == 0:
                    return "No limits found with high utilization (above 95%) in the latest data snapshot."
                elif row_count == 1:
                    return f"Based on the query results, I found **1 limit with high utilization** (above 95%).\n\nThis limit is near or at capacity and should be monitored closely for potential breach risk."
                else:
                    return f"""Based on the query results, I found **{row_count} limits with high utilization** (above 95%). Here are the key insights:

## Summary
- **Total high-utilization limits**: {row_count}
- **Threshold**: Utilization > 95% (near or at capacity)
- **Data snapshot**: Latest available date

## Analysis
These limits are operating near or at capacity and require close monitoring. High utilization indicates that these limits may breach with small exposure increases.

## Recommendations
- Review limits at or near 100% utilization immediately for potential breach risk
- Monitor limits above 95% utilization closely
- Consider limit increases for frequently utilized limits in active trading desks
- Review exposure trends to identify patterns in high utilization

The data shows {row_count} limit(s) requiring attention based on the high utilization threshold."""
        
        # For other queries in demo mode, try to extract row count and generate appropriate response
        if response_format == "json":
            return json.dumps({
                "sql": "SELECT * FROM limits_data LIMIT 10",
                "reasoning": "Demo mode: Returning sample query",
                "plan_quality": "medium",
                "confidence": 0.7
            })
        else:
            # Try to extract row count from prompt for generic queries too
            row_count = 0
            try:
                import re
                match = re.search(r'"row_count":\s*(\d+)', user_prompt)
                if match:
                    row_count = int(match.group(1))
            except:
                pass
            
            if row_count > 0:
                return f"Demo mode is active. Query executed successfully and returned {row_count} result(s). This is a mock response. For detailed analysis, please configure ANTHROPIC_API_KEY."
            else:
                return "Demo mode is active. This is a mock response. For full functionality, please configure ANTHROPIC_API_KEY."
    
    def get_info(self) -> Dict[str, Any]:
        """Get LLM configuration info"""
        return {
            "available": self.is_available(),
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "demo_mode": self.demo_mode
        }


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client"""
    global _llm_client
    # Re-initialize if not available and API key is now present
    if _llm_client is None or (not _llm_client.is_available() and os.getenv("ANTHROPIC_API_KEY")):
        _llm_client = LLMClient()
    return _llm_client


def is_llm_available() -> bool:
    """Quick check if LLM is available"""
    return get_llm_client().is_available()


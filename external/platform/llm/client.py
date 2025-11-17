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
        
        print("⚠️  No LLM configured. Set ANTHROPIC_API_KEY in .env")
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.client is not None
    
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
    
    def get_info(self) -> Dict[str, Any]:
        """Get LLM configuration info"""
        return {
            "available": self.is_available(),
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
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


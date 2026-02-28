"""
Gemini Context Caching - Transcript Analysis V2
================================================
Uses Gemini's Context Caching feature for 90% cost reduction
when analyzing the same transcript multiple times.

Features:
- Cache transcript content for repeated queries
- 90% cost reduction for cached content
- Automatic cache invalidation after TTL
"""

import os
import hashlib
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GeminiContextCache:
    """
    Manages cached contexts for Gemini API calls.
    
    Context Caching allows reusing uploaded content (like transcripts)
    across multiple API calls at reduced cost.
    
    Pricing:
    - Regular input: $0.50 per 1M tokens
    - Cached input: $0.05 per 1M tokens (90% savings!)
    
    Usage:
        cache = GeminiContextCache()
        cache_id = cache.create_cache(transcript_text)
        
        # Use cached context in multiple queries
        result1 = cache.query(cache_id, "Find viral moments")
        result2 = cache.query(cache_id, "Identify key themes")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview",
        default_ttl_hours: int = 1
    ):
        """
        Initialize context cache manager.
        
        Args:
            api_key: Gemini API key
            model_name: Model to use
            default_ttl_hours: Default TTL for cached content
        """
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-genai package required. Run: pip install google-genai")
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.default_ttl = timedelta(hours=default_ttl_hours)
        
        # Local cache tracking
        self._active_caches: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"GeminiContextCache initialized (TTL: {default_ttl_hours}h)")
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for cache key."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def create_cache(
        self,
        transcript: str,
        system_instruction: Optional[str] = None,
        ttl_hours: Optional[int] = None
    ) -> str:
        """
        Create a cached context for a transcript.
        
        Args:
            transcript: The full transcript text
            system_instruction: Optional system prompt
            ttl_hours: Cache TTL in hours
        
        Returns:
            Cache ID for future queries
        """
        content_hash = self._hash_content(transcript)
        
        # Check if already cached
        if content_hash in self._active_caches:
            cache_info = self._active_caches[content_hash]
            if datetime.now() < cache_info["expires"]:
                logger.info(f"Using existing cache: {content_hash}")
                return content_hash
        
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
        
        try:
            # Create cached content
            # Note: The exact API may vary - this follows the expected pattern
            cached_content = self.client.caches.create(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": transcript}]}],
                system_instruction=system_instruction or self._default_system_instruction(),
                ttl=f"{int(ttl.total_seconds())}s"
            )
            
            self._active_caches[content_hash] = {
                "cache_name": cached_content.name,
                "created": datetime.now(),
                "expires": datetime.now() + ttl,
                "token_count": len(transcript) // 4  # Rough estimate
            }
            
            logger.info(f"Created cache: {content_hash} (expires: {self._active_caches[content_hash]['expires']})")
            return content_hash
            
        except Exception as e:
            logger.warning(f"Context caching not available: {e}. Falling back to regular queries.")
            # Store None to indicate no caching available
            self._active_caches[content_hash] = {
                "cache_name": None,
                "transcript": transcript,
                "created": datetime.now(),
                "expires": datetime.now() + ttl
            }
            return content_hash
    
    def _default_system_instruction(self) -> str:
        """Default system instruction for video analysis."""
        return """You are an expert video content analyst specializing in viral short-form content.
Your task is to analyze video transcripts and identify:
1. Viral-worthy segments (30-90 seconds)
2. Compelling hooks (attention-grabbing openings)
3. Key themes and target audience
4. Content quality and editing suggestions

Always provide specific timestamps and actionable insights."""
    
    def query(
        self,
        cache_id: str,
        query: str,
        response_schema: Optional[type] = None,
        temperature: float = 1.0
    ) -> str:
        """
        Query the cached context.
        
        Args:
            cache_id: Cache ID from create_cache()
            query: The query to run against cached content
            response_schema: Optional Pydantic schema for response
            temperature: Generation temperature
        
        Returns:
            Generated response text
        """
        if cache_id not in self._active_caches:
            raise ValueError(f"Unknown cache ID: {cache_id}")
        
        cache_info = self._active_caches[cache_id]
        
        # Check if cache has expired
        if datetime.now() >= cache_info["expires"]:
            logger.warning(f"Cache {cache_id} expired")
            del self._active_caches[cache_id]
            raise ValueError(f"Cache {cache_id} has expired")
        
        config_params = {
            "temperature": temperature,
            "max_output_tokens": 8192
        }
        
        if response_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_schema
        
        try:
            if cache_info.get("cache_name"):
                # Use cached content
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[{"role": "user", "parts": [{"text": query}]}],
                    cached_content=cache_info["cache_name"],
                    config=config_params
                )
            else:
                # Fallback: include transcript in request
                transcript = cache_info.get("transcript", "")
                full_prompt = f"""TRANSCRIPT:
{transcript}

QUERY:
{query}"""
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=config_params
                )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def invalidate(self, cache_id: str):
        """Manually invalidate a cache."""
        if cache_id in self._active_caches:
            cache_info = self._active_caches[cache_id]
            
            # Try to delete from API
            if cache_info.get("cache_name"):
                try:
                    self.client.caches.delete(name=cache_info["cache_name"])
                except:
                    pass
            
            del self._active_caches[cache_id]
            logger.info(f"Invalidated cache: {cache_id}")
    
    def get_savings_estimate(self, cache_id: str, num_queries: int) -> Dict[str, float]:
        """
        Estimate cost savings from using cached context.
        
        Args:
            cache_id: Cache ID
            num_queries: Expected number of queries
        
        Returns:
            Dict with cost estimates
        """
        if cache_id not in self._active_caches:
            return {"error": "Unknown cache"}
        
        cache_info = self._active_caches[cache_id]
        token_count = cache_info.get("token_count", 0)
        
        # Pricing per 1M tokens
        regular_price = 0.50
        cached_price = 0.05
        
        # Cost for first query (cache creation)
        creation_cost = (token_count / 1_000_000) * regular_price
        
        # Cost for subsequent queries
        cached_query_cost = (token_count / 1_000_000) * cached_price
        
        total_cached = creation_cost + (cached_query_cost * (num_queries - 1))
        total_regular = (token_count / 1_000_000) * regular_price * num_queries
        
        return {
            "token_count": token_count,
            "regular_cost": round(total_regular, 4),
            "cached_cost": round(total_cached, 4),
            "savings": round(total_regular - total_cached, 4),
            "savings_percent": round((1 - total_cached / total_regular) * 100, 1) if total_regular > 0 else 0
        }


# Convenience function
def create_transcript_cache(transcript: str) -> GeminiContextCache:
    """Create a cache for a transcript and return the manager."""
    cache = GeminiContextCache()
    cache.create_cache(transcript)
    return cache

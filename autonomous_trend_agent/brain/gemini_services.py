# brain/gemini_services.py
"""
Unified Gemini Services (Modern Stack)
Provides AI-powered enhancements: metadata, hook validation, QC, and more.
Uses google-genai SDK (v1.0+) via GeminiClientV2 pattern.
"""
import os
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential,
    retry_if_exception_type
)
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR STRICT RESPONSE VALIDATION
# =============================================================================

class Metadata(BaseModel):
    """Generated metadata for a short."""
    title: str = Field(description="Catchy title under 60 chars with emoji")
    description: str = Field(description="SEO description 150-200 chars")
    tags: List[str] = Field(default_factory=list, description="5 relevant tags")
    hashtags: List[str] = Field(default_factory=list, description="3-5 hashtags")


class HookScore(BaseModel):
    """Validation of the hook (first 3 seconds)."""
    score: int = Field(ge=0, le=100, description="Hook effectiveness 0-100")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class QCResult(BaseModel):
    """Quality control check results."""
    passed: bool = Field(description="Whether the short passes QC")
    score: int = Field(ge=0, le=100, description="Quality score 0-100")
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ThumbnailSuggestion(BaseModel):
    """Suggested thumbnail for the short."""
    timestamp: float = Field(description="Best frame timestamp in seconds")
    description: str = Field(description="Description of the visual")
    text_overlay: str = Field(description="Suggested text to overlay")


class ContentWarning(BaseModel):
    """Content warning detection."""
    category: str = Field(description="profanity, violence, sensitive, etc.")
    severity: str = Field(description="low, medium, high")
    timestamp: Optional[float] = None


class ShortEnhancement(BaseModel):
    """Complete enhancement result for a short video."""
    metadata: Metadata
    hook_score: HookScore
    qc_result: QCResult
    caption_style: str = Field(default="tiktok", description="tiktok, hormozi, mrbeast, minimal, neon")
    thumbnail: ThumbnailSuggestion
    content_warnings: List[ContentWarning] = Field(default_factory=list)


# =============================================================================
# GEMINI CLIENT (Service Layer)
# =============================================================================

class GeminiServices:
    """
    Unified Gemini Service using google.genai SDK.
    Replaces legacy GeminiServices.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview"
    ):
        # Import the new SDK to ensure availability
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        
        # Get API key
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Respect env var override
        self.model_name = os.getenv("GEMINI_MODEL_NAME", model_name)
        
        logger.info(f"[GeminiServices] Initialized with model: {self.model_name}")

    def enhance_short(
        self,
        transcript_text: str,
        hook_text: str,
        theme: str = "",
        audience: str = "",
        duration_seconds: float = 60
    ) -> ShortEnhancement:
        """
        Generate all AI enhancements for a short in a single API call.
        """
        prompt = f"""Analyze this short-form video and provide complete AI enhancements.

TRANSCRIPT:
{transcript_text}

HOOK (first 3 seconds): {hook_text}
THEME: {theme}
AUDIENCE: {audience}
DURATION: {duration_seconds:.0f} seconds

Provide ALL of the following as structured JSON:

1. METADATA: Generate title (catchy, <60 chars, with emoji), description (SEO, 150-200 chars), 
   5 tags, and 3-5 hashtags.

2. HOOK_SCORE: Rate the hook 0-100, list strengths, weaknesses, and improvement suggestions.

3. QC_RESULT: Check for quality issues (abrupt cuts, incoherent content, mid-sentence endings).
   Score 0-100 and list any issues.

4. CAPTION_STYLE: Recommend one of: tiktok, hormozi, mrbeast, minimal, neon

5. THUMBNAIL: Suggest the best timestamp for thumbnail, describe the visual, suggest text overlay.

6. CONTENT_WARNINGS: Detect any profanity, violence, or sensitive content with severity."""

        try:
            response = self._generate(prompt, response_schema=ShortEnhancement)
            # The SDK might return parsed object or text depending on version/config
            # With typed config, it usually returns a structure or we parse JSON.
            # Safe bet: parse JSON from text.
            data = json.loads(response)
            return ShortEnhancement(**data)
        except Exception as e:
            logger.error(f"[GeminiServices] Enhancement failed: {e}")
            return self._default_enhancement()

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def _generate(
        self, 
        prompt: str, 
        response_schema: Optional[type] = None,
        temperature: float = 1.0
    ) -> str:
        """Core generation method with retry."""
        config_params = {
            "temperature": temperature,
            "max_output_tokens": 8192
        }
        
        if response_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_schema
        
        config = self.types.GenerateContentConfig(**config_params)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return response.text

    def _default_enhancement(self) -> ShortEnhancement:
        """Fallback enhancement."""
        return ShortEnhancement(
            metadata=Metadata(title="Must Watch", description="", tags=[], hashtags=[]),
            hook_score=HookScore(score=50),
            qc_result=QCResult(passed=True, score=70),
            caption_style="tiktok",
            thumbnail=ThumbnailSuggestion(timestamp=0, description="", text_overlay=""),
            content_warnings=[]
        )
    
    # --- Individual methods for backward compatibility if needed ---
    
    def validate_hook(self, hook_text: str) -> HookScore:
        prompt = f"Rate this hook: '{hook_text}'"
        try:
            res = self._generate(prompt, response_schema=HookScore)
            return HookScore(**json.loads(res))
        except:
            return HookScore(score=0)

    def save_enhancement(self, enhancement: ShortEnhancement, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhancement.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"[GeminiServices] Saved to {output_path}")

from pathlib import Path

if __name__ == "__main__":
    pass

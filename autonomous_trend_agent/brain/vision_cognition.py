"""
Qwen3-VL Vision Cognition - V2
==============================
Advanced video understanding using Qwen3-VL-8B-Instruct.

Features:
- 128K token context window (process long videos)
- Int4 quantization (~5.5GB VRAM)
- "Thinking" mode for complex reasoning
- Scene understanding and viral moment detection

Architecture:
- Uses Hub-and-Spoke: Backbone stays loaded, task heads swap
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import base64
from io import BytesIO

import torch

logger = logging.getLogger(__name__)


@dataclass
class ViralMoment:
    """A detected viral-worthy moment."""
    start_time: float
    end_time: float
    description: str
    hook: str  # Opening line suggestion
    viral_score: int  # 0-100
    reason: str
    editing_notes: str = ""


@dataclass
class SceneAnalysis:
    """Analysis of a single scene/frame."""
    timestamp: float
    description: str
    subjects: List[str]
    emotions: List[str]
    actions: List[str]
    visual_quality: int  # 0-100
    is_key_frame: bool = False


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result."""
    summary: str
    theme: str
    target_audience: str
    scenes: List[SceneAnalysis]
    viral_moments: List[ViralMoment]
    content_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "theme": self.theme,
            "target_audience": self.target_audience,
            "scenes": [
                {
                    "timestamp": s.timestamp,
                    "description": s.description,
                    "subjects": s.subjects,
                    "emotions": s.emotions,
                    "actions": s.actions,
                    "visual_quality": s.visual_quality,
                    "is_key_frame": s.is_key_frame
                }
                for s in self.scenes
            ],
            "viral_moments": [
                {
                    "start_time": m.start_time,
                    "end_time": m.end_time,
                    "description": m.description,
                    "hook": m.hook,
                    "viral_score": m.viral_score,
                    "reason": m.reason,
                    "editing_notes": m.editing_notes
                }
                for m in self.viral_moments
            ],
            "content_warnings": self.content_warnings
        }


class Qwen3VLCognition:
    """
    Vision-Language understanding using Qwen3-VL-8B-Instruct.
    
    Qwen3-VL is the state-of-the-art VLM with:
    - 128K context window (process full videos)
    - Native video understanding
    - "Thinking" mode for complex reasoning
    
    Uses Int4 quantization to fit in ~5.5GB VRAM.
    
    Usage:
        vlm = Qwen3VLCognition()
        result = vlm.analyze_video_frames(frames, fps=30)
        moments = result.viral_moments
    """
    
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    
    def __init__(
        self,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_context_length: int = 32768,  # Conservative for VRAM
        enable_thinking: bool = True
    ):
        """
        Initialize Qwen3-VL.
        
        Args:
            device: "cuda" or "cpu"
            load_in_4bit: Use Int4 quantization (~5.5GB VRAM)
            max_context_length: Max tokens in context
            enable_thinking: Enable "thinking" reasoning mode
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.load_in_4bit = load_in_4bit
        self.max_context_length = max_context_length
        self.enable_thinking = enable_thinking
        
        self._model = None
        self._processor = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            from transformers import BitsAndBytesConfig
            
            logger.info(f"Loading Qwen3-VL-8B-Instruct (4bit={self.load_in_4bit})...")
            
            # Configure quantization
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self._loaded = True
            
            # Log VRAM usage
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Qwen3-VL loaded. VRAM: {vram_gb:.2f}GB")
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Run: pip install transformers accelerate bitsandbytes qwen-vl-utils"
            )
    
    def _prepare_image(self, image: Union[str, bytes, "PIL.Image.Image"]) -> str:
        """Convert image to base64 for model input."""
        from PIL import Image
        
        if isinstance(image, str):
            # File path
            with open(image, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image, bytes):
            img_bytes = image
        else:
            # PIL Image
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode("utf-8")
    
    def analyze_frame(
        self,
        frame: Union[str, bytes, torch.Tensor],
        prompt: str = "Describe this video frame in detail.",
        timestamp: float = 0.0
    ) -> SceneAnalysis:
        """
        Analyze a single frame.
        
        Args:
            frame: Image path, bytes, or tensor
            prompt: Analysis prompt
            timestamp: Frame timestamp
        
        Returns:
            SceneAnalysis with description and metadata
        """
        self._ensure_loaded()
        
        # Convert tensor to PIL if needed
        if isinstance(frame, torch.Tensor):
            from PIL import Image
            import numpy as np
            
            if frame.dim() == 3:
                if frame.shape[0] == 3:  # CHW -> HWC
                    frame = frame.permute(1, 2, 0)
                frame = frame.cpu().numpy().astype(np.uint8)
            frame = Image.fromarray(frame)
        
        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process and generate
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._processor(
            text=[text],
            images=[frame],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response into SceneAnalysis
        return SceneAnalysis(
            timestamp=timestamp,
            description=response,
            subjects=[],  # Could parse from response
            emotions=[],
            actions=[],
            visual_quality=80,
            is_key_frame=False
        )
    
    def analyze_video_frames(
        self,
        frames: List[torch.Tensor],
        fps: float = 30.0,
        transcript: Optional[str] = None,
        num_shorts: int = 3
    ) -> VideoAnalysisResult:
        """
        Analyze video frames to find viral moments.
        
        Args:
            frames: List of frame tensors (sampled from video)
            fps: Original video FPS
            transcript: Optional transcript for context
            num_shorts: Number of viral moments to find
        
        Returns:
            VideoAnalysisResult with scenes and viral moments
        """
        self._ensure_loaded()
        
        # Sample frames uniformly
        num_sample = min(len(frames), 8)  # Limit to 8 frames for VRAM
        indices = [i * len(frames) // num_sample for i in range(num_sample)]
        sampled_frames = [frames[i] for i in indices]
        timestamps = [i / fps for i in indices]
        
        logger.info(f"Analyzing {len(sampled_frames)} frames from video")
        
        # Build multi-image prompt
        prompt = self._build_analysis_prompt(transcript, num_shorts)
        
        # Convert frames to PIL
        from PIL import Image
        import numpy as np
        
        pil_frames = []
        for frame in sampled_frames:
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0)
                frame = frame.cpu().numpy().astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))
        
        # Build conversation with multiple images
        content = []
        for i, img in enumerate(pil_frames):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[Frame {i+1} at {timestamps[i]:.1f}s]"})
        
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Enable thinking mode if configured
        if self.enable_thinking:
            messages[0]["content"].insert(0, {
                "type": "text",
                "text": "/think"  # Qwen3 thinking mode trigger
            })
        
        # Process and generate
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._processor(
            text=[text],
            images=pil_frames,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True
            )
        
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        return self._parse_analysis_response(response, timestamps)
    
    def _build_analysis_prompt(
        self,
        transcript: Optional[str],
        num_shorts: int
    ) -> str:
        """Build the analysis prompt."""
        prompt = f"""Analyze these video frames to identify {num_shorts} viral-worthy moments for short-form content (30-90 seconds each).

For each moment, provide:
1. Start and end timestamps
2. A compelling hook (the opening line that grabs attention)
3. Viral score (0-100)
4. Why it would go viral
5. Editing suggestions

Also identify:
- Overall theme and message
- Target audience
- Any content warnings"""

        if transcript:
            prompt += f"""

TRANSCRIPT (for context):
{transcript[:2000]}..."""  # Limit transcript length

        prompt += """

Return your analysis as structured JSON:
{
  "summary": "...",
  "theme": "...",
  "target_audience": "...",
  "viral_moments": [
    {
      "start_time": 0.0,
      "end_time": 60.0,
      "description": "...",
      "hook": "...",
      "viral_score": 85,
      "reason": "...",
      "editing_notes": "..."
    }
  ],
  "content_warnings": []
}"""
        return prompt
    
    def _parse_analysis_response(
        self,
        response: str,
        timestamps: List[float]
    ) -> VideoAnalysisResult:
        """Parse model response into VideoAnalysisResult."""
        # Try to extract JSON from response
        try:
            # Find JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                viral_moments = [
                    ViralMoment(
                        start_time=m.get("start_time", 0),
                        end_time=m.get("end_time", 60),
                        description=m.get("description", ""),
                        hook=m.get("hook", ""),
                        viral_score=m.get("viral_score", 50),
                        reason=m.get("reason", ""),
                        editing_notes=m.get("editing_notes", "")
                    )
                    for m in data.get("viral_moments", [])
                ]
                
                return VideoAnalysisResult(
                    summary=data.get("summary", ""),
                    theme=data.get("theme", ""),
                    target_audience=data.get("target_audience", "General"),
                    scenes=[],
                    viral_moments=viral_moments,
                    content_warnings=data.get("content_warnings", [])
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response")
        
        # Fallback: return basic result
        return VideoAnalysisResult(
            summary=response[:500],
            theme="Unknown",
            target_audience="General",
            scenes=[],
            viral_moments=[],
            content_warnings=[]
        )
    
    def unload(self):
        """Unload model to free VRAM."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        logger.info("Qwen3-VL unloaded")


# Convenience function
def analyze_video(
    video_path: str,
    transcript: Optional[str] = None,
    num_shorts: int = 3
) -> VideoAnalysisResult:
    """
    Analyze video for viral moments.
    
    Args:
        video_path: Path to video file
        transcript: Optional transcript text
        num_shorts: Number of viral moments to find
    
    Returns:
        VideoAnalysisResult with viral moments
    """
    from ..core.gpu_pipeline import GPUVideoDecoder
    
    # Extract frames
    decoder = GPUVideoDecoder(video_path)
    info = decoder.info
    
    # Sample ~1 frame per 30 seconds
    sample_interval = int(info.fps * 30)
    frames = []
    
    for i, frame in enumerate(decoder):
        if i % sample_interval == 0:
            frames.append(frame.clone())
        if len(frames) >= 20:  # Max 20 samples
            break
    
    decoder.close()
    
    # Analyze
    vlm = Qwen3VLCognition()
    try:
        result = vlm.analyze_video_frames(frames, info.fps, transcript, num_shorts)
        return result
    finally:
        vlm.unload()

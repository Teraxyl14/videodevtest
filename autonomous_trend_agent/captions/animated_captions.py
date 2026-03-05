"""
Animated Caption Engine
Creates viral-style word-by-word highlight captions like TikTok/Reels

Features:
- Word-by-word highlight animation with pop scale (125%)
- Multiple style presets (TikTok, Hormozi, MrBeast)
- Fail-fast font loading (no silent fallback to 11px bitmap)
- Platform safe zone (1010x1420px centered on 1080x1920)
- O(1) Pillow stroke_width for outlines (not O(n²) pixel loop)
- -50ms Negative Perceptual Offset (NPO)
- GPU-accelerated rendering via Kornia
- ASS subtitle generation for complex animations
"""

import os
import json
import subprocess
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
# import kornia  <-- Moved to lazy import

# =============================================================================
# Font Discovery — Ordered search paths for Docker + local development
# =============================================================================
FONT_SEARCH_PATHS = [
    # 1. Bundled custom fonts (downloaded at Docker build time)
    "/usr/local/share/fonts/custom/Montserrat-Black.ttf",
    # 2. apt-get installed bold fonts (Docker: fonts-freefont-ttf, fonts-liberation)
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # 3. System fonts (msttcorefonts if installed)
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
    # 4. Windows paths (for local dev outside Docker)
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
]

# Font name -> fallback path mapping for specific style presets
FONT_FALLBACK_MAP = {
    "Impact": ["FreeSansBold.ttf", "Impact.ttf", "impact.ttf"],
    "Arial Black": ["LiberationSans-Bold.ttf", "Arial_Black.ttf", "ariblk.ttf"],
    "Montserrat Black": ["Montserrat-Black.ttf"],
    "Montserrat-Black": ["Montserrat-Black.ttf"],
    "FreeSans Bold": ["FreeSansBold.ttf"],
    "Liberation Sans": ["LiberationSans-Bold.ttf"],
}

# =============================================================================
# Platform Safe Zone — 1010x1420px centered on 1080x1920 canvas
# Avoids platform UI overlays (comments, likes, description)
# =============================================================================
SAFE_ZONE = {
    "left": 35,      # (1080 - 1010) / 2
    "right": 1045,
    "top": 250,      # (1920 - 1420) / 2
    "bottom": 1670,
    "width": 1010,
    "height": 1420,
}


def _discover_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    """
    Discover and load a font with FAIL-FAST behavior.
    No silent fallback to 11px bitmap — errors loudly if no usable font found.
    """
    # Try exact font name as path first
    for suffix in [".ttf", ""]:
        try:
            return ImageFont.truetype(f"{font_name}{suffix}", font_size)
        except (OSError, IOError):
            pass
    
    # Try name-based fallback paths
    fallback_names = FONT_FALLBACK_MAP.get(font_name, [f"{font_name}.ttf"])
    search_dirs = [
        "/usr/local/share/fonts/custom",
        "/usr/share/fonts/truetype/msttcorefonts",
        "/usr/share/fonts/truetype/dejavu",
        "C:/Windows/Fonts",
    ]
    for d in search_dirs:
        for fname in fallback_names:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                print(f"[AnimatedCaptions] Found font: {path}")
                return ImageFont.truetype(path, font_size)
    
    # Try all search paths as absolute last resort
    for path in FONT_SEARCH_PATHS:
        if os.path.exists(path):
            print(f"[AnimatedCaptions] WARNING: Using fallback font {path} (wanted '{font_name}')")
            return ImageFont.truetype(path, font_size)
    
    # FAIL-FAST: Do NOT silently return load_default() which gives 11px bitmap
    raise FileNotFoundError(
        f"CRITICAL: No usable font found for '{font_name}'. "
        f"Searched: {FONT_SEARCH_PATHS}. "
        f"Install fonts via: apt-get install fonts-dejavu-core OR "
        f"place .ttf files in assets/fonts/ and rebuild Docker image."
    )



@dataclass
class CaptionStyle:
    """Caption styling configuration"""
    name: str
    font_name: str = "Arial"
    font_size: int = 96                 # Default: scaled down for 1080x1920 from 640x360 source
    primary_color: str = "FFFFFF"      # White
    highlight_color: str = "FFD700"    # Gold
    outline_color: str = "000000"      # Black
    outline_width: int = 5             # Scaled outline
    shadow_color: str = "000000"
    shadow_depth: int = 4
    alignment: int = 2                  # Bottom center
    margin_v: int = 150                 # Large vertical margin (v2)
    bold: bool = True
    animation: str = "highlight"        # highlight, pop, wave
    words_per_line: int = 3
    npo: int = -50                      # Negative Perceptual Offset (ms)


# Preset styles
CAPTION_STYLES = {
    "tiktok": CaptionStyle(
        name="tiktok",
        font_name="Impact",
        font_size=70,
        primary_color="FFFFFF",
        highlight_color="00FF00",  # Green highlight
        outline_width=6,
        shadow_depth=4,
        bold=True,
        animation="highlight",
        words_per_line=2,
        margin_v=180,
        npo=-50
    ),
    "hormozi": CaptionStyle(
        name="hormozi",  
        font_name="Arial Black",
        font_size=80,
        primary_color="FFFFFF",
        highlight_color="FFFF00",  # Yellow
        outline_color="000000",
        outline_width=6,
        shadow_depth=5,
        bold=True,
        animation="pop",
        words_per_line=1,  # One word at a time
        margin_v=200,
        npo=-50
    ),
    "mrbeast": CaptionStyle(
        name="mrbeast",
        font_name="Impact",
        font_size=70,
        primary_color="FFFFFF",
        highlight_color="FF0000",  # Red
        outline_color="000000",
        outline_width=6,
        shadow_depth=4,
        bold=True,
        animation="highlight",
        words_per_line=3,
        margin_v=180,
        npo=-50
    ),
    "minimal": CaptionStyle(
        name="minimal",
        font_name="Helvetica",
        font_size=64,
        primary_color="FFFFFF",
        highlight_color="AAAAFF",  # Light blue
        outline_width=4,
        shadow_depth=3,
        bold=False,
        animation="highlight",
        words_per_line=4,
        margin_v=150,
        npo=0  # No offset for minimal
    ),
    "neon": CaptionStyle(
        name="neon",
        font_name="Arial",
        font_size=65,
        primary_color="00FFFF",   # Cyan
        highlight_color="FF00FF", # Magenta
        outline_color="000000",
        outline_width=4,
        shadow_depth=6,
        bold=True,
        animation="wave",
        words_per_line=2,
        margin_v=180,
        npo=-50
    )
}


class AnimatedCaptionEngine:
    """
    Creates animated captions with word-by-word highlighting.
    Outputs ASS (Advanced SubStation Alpha) format for complex animations.
    """
    
    def __init__(self, style: str = "tiktok", device: str = "cuda"):
        """
        Initialize with a style preset.
        
        Args:
            style: Preset name or CaptionStyle object
            device: 'cuda' or 'cpu'
        """
        self.device = device
        if isinstance(style, str):
            self.style = CAPTION_STYLES.get(style.lower(), CAPTION_STYLES["tiktok"])
        else:
            self.style = style
        
        self._line_cache = None
        self._source_words_id = None
        
        print(f"[AnimatedCaptions] Style: {self.style.name}")
        
    def _extract_words(self, transcript_data: Dict) -> List[Dict]:
        """Extract word-level timestamps from transcript data blob."""
        words = []
        if not transcript_data:
            return []
            
        # Parakeet/NeMo format
        if "words" in transcript_data and isinstance(transcript_data["words"], list):
            words = transcript_data["words"]
        elif "result" in transcript_data and "words" in transcript_data["result"]:
            words = transcript_data["result"]["words"]
        # Whisper format
        elif "segments" in transcript_data:
            for seg in transcript_data["segments"]:
                if "words" in seg:
                    words.extend(seg["words"])
                else:
                    # Fallback to segment-level if word-level missing
                    words.append({
                        "word": seg.get("text", ""),
                        "start_time": seg.get("start", 0),
                        "end_time": seg.get("end", 0)
                    })
        # Already a list
        elif isinstance(transcript_data, list):
            words = transcript_data
            
        # Standardize keys (some use 'start', some 'start_time')
        standardized = []
        for w in words:
            standardized.append({
                "word": w.get("word", w.get("text", "")),
                "start_time": float(w.get("start_time", w.get("start", 0))),
                "end_time": float(w.get("end_time", w.get("end", 0)))
            })
        return standardized

    def render_to_tensor(
        self,
        words_or_transcript: List[Dict],
        timestamp: float,
        width: int,
        height: int
    ) -> Optional[torch.Tensor]:
        """
        Render captions for a specific timestamp to a GPU RGBA tensor.
        
        Features:
        - Fail-fast font loading (no silent 11px bitmap fallback)
        - O(1) Pillow stroke_width for outlines (not O(n²) pixel loop)
        - Platform safe zone positioning (1010x1420px centered)
        - Pop animation: active word rendered at 125% scale
        - -50ms Negative Perceptual Offset (NPO)
        
        Args:
            words_or_transcript: List of word dicts or full transcript blob
            timestamp: Current time in seconds
            width: Canvas width
            height: Canvas height
            
        Returns:
            (4, H, W) float32 tensor on GPU or None if no text
        """
        # Ensure we have words
        if isinstance(words_or_transcript, dict):
            words = self._extract_words(words_or_transcript)
        else:
            words = words_or_transcript
            
        if not words:
            return None
            
        # Lazy Group/Cache lines
        words_id = id(words_or_transcript)
        if self._source_words_id != words_id:
            self._line_cache = self._group_into_lines(words)
            self._source_words_id = words_id
        
        # Apply Negative Perceptual Offset (-50ms): show captions slightly early
        npo_seconds = self.style.npo / 1000.0  # -50ms -> -0.05s
        adjusted_timestamp = timestamp - npo_seconds  # Subtracting negative = adding 50ms
            
        # Find active line in cache using NPO-adjusted timestamp
        active_line = None
        for line in self._line_cache:
            if line["start_time"] <= adjusted_timestamp <= line["end_time"]:
                active_line = line
                break
        
        if not active_line:
            return None
            
        # Lazy-load font (fail-fast, cached after first call)
        if not hasattr(self, '_font') or self._font is None:
            self._font = _discover_font(self.style.font_name, self.style.font_size)
            print(f"[AnimatedCaptions] Font loaded: {self.style.font_name} @ {self.style.font_size}px")
        font = self._font
        
        # For pop animation, also load a scaled font (115%)
        if not hasattr(self, '_font_pop') or self._font_pop is None:
            pop_size = int(self.style.font_size * 1.15)
            self._font_pop = _discover_font(self.style.font_name, pop_size)
        font_pop = self._font_pop
        
        # Create canvas
        canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Calculate safe zone Y position
        # Caption sits in bottom portion of safe zone
        # Safe zone bottom = 1670 for 1080x1920
        safe_bottom = min(SAFE_ZONE["bottom"], height - 250)
        safe_left = SAFE_ZONE["left"] if width >= 1080 else 10
        safe_right = SAFE_ZONE["right"] if width >= 1080 else width - 10
        safe_width = safe_right - safe_left
        
        # Y position: place text near bottom of safe zone with margin
        y = safe_bottom - self.style.margin_v
        
        # Measure total line width to center within safe zone
        full_text = active_line["text"]
        total_w = draw.textlength(full_text, font=font)
        
        # If text overflows safe width, it's okay — we just center it
        line_center_x = safe_left + safe_width // 2
        start_x = int(line_center_x - total_w / 2)
        
        curr_x = start_x
        stroke_w = self.style.outline_width
        outline_fill = f"#{self.style.outline_color}"
        
        for word_info in active_line["words"]:
            # Add an extra space to the right to help keep words visually separated
            w_text = word_info["word"] + "  "
            is_active = word_info["start_time"] <= adjusted_timestamp <= word_info["end_time"]
            
            if is_active:
                # === POP ANIMATION: Active word at 115% scale ===
                color = f"#{self.style.highlight_color}"
                
                # Measure normal-size width (for cursor advance)
                w_width_normal = draw.textlength(w_text, font=font)
                
                # Measure pop-size width
                w_width_pop = draw.textlength(w_text, font=font_pop)
                
                # Center the popped word over its normal position
                pop_x = int(curr_x + w_width_normal / 2 - w_width_pop / 2)
                
                # Shift Y up slightly for pop (so it expands upward from baseline)
                pop_y_offset = int((self.style.font_size * 0.15) / 2)  # Half of size difference
                pop_y = y - pop_y_offset
                
                # Draw popped word with stroke_width (O(1) C-level outline)
                if stroke_w > 0:
                    draw.text(
                        (pop_x, pop_y), w_text, font=font_pop,
                        fill=color, stroke_width=stroke_w, stroke_fill=outline_fill
                    )
                else:
                    draw.text((pop_x, pop_y), w_text, font=font_pop, fill=color)
                
                curr_x += w_width_normal
            else:
                # === Normal word rendering ===
                color = f"#{self.style.primary_color}"
                w_width = draw.textlength(w_text, font=font)
                
                # Draw with stroke_width (O(1) C-level outline)
                if stroke_w > 0:
                    draw.text(
                        (curr_x, y), w_text, font=font,
                        fill=color, stroke_width=stroke_w, stroke_fill=outline_fill
                    )
                else:
                    draw.text((curr_x, y), w_text, font=font, fill=color)
                
                curr_x += w_width
            
        # Convert to Tensor: (H, W, 4) -> (4, H, W)
        img_np = np.array(canvas)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    
    def generate_ass(
        self,
        words: List[Dict],
        output_path: str,
        video_width: int = 1080,
        video_height: int = 1920
    ) -> str:
        """
        Generate ASS subtitle file with word-by-word animation.
        
        Args:
            words: List of {"word": str, "start_time": float, "end_time": float}
            output_path: Path to save .ass file
            video_width: Video width for positioning
            video_height: Video height for positioning
            
        Returns:
            Path to generated ASS file
        """
        # Group words into lines
        lines = self._group_into_lines(words)
        
        # Generate ASS content
        ass_content = self._generate_ass_header(video_width, video_height)
        ass_content += "\n[Events]\n"
        ass_content += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        
        for line in lines:
            if self.style.animation == "highlight":
                ass_content += self._generate_highlight_events(line)
            elif self.style.animation == "pop":
                ass_content += self._generate_pop_events(line)
            elif self.style.animation == "wave":
                ass_content += self._generate_wave_events(line)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        
        print(f"[AnimatedCaptions] Generated: {output_path} ({len(lines)} lines)")
        return output_path
    
    def _generate_ass_header(self, width: int, height: int) -> str:
        """Generate ASS file header with styles"""
        s = self.style
        
        # Convert hex colors to ASS format (BGR with alpha prefix)
        primary = self._hex_to_ass_color(s.primary_color)
        highlight = self._hex_to_ass_color(s.highlight_color)
        outline = self._hex_to_ass_color(s.outline_color)
        shadow = self._hex_to_ass_color(s.shadow_color)
        
        header = f"""[Script Info]
Title: Animated Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{s.font_name},{s.font_size},{primary},{highlight},{outline},{shadow},{int(s.bold)},0,0,0,100,100,0,0,1,{s.outline_width},{s.shadow_depth},{s.alignment},10,10,{s.margin_v},1
Style: Highlight,{s.font_name},{int(s.font_size * 1.1)},{highlight},{primary},{outline},{shadow},{int(s.bold)},0,0,0,100,100,0,0,1,{s.outline_width},{s.shadow_depth},{s.alignment},10,10,{s.margin_v},1
"""
        return header
    
    def _hex_to_ass_color(self, hex_color: str) -> str:
        """Convert hex color to ASS format (&HBBGGRR&)"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"&H00{b:02X}{g:02X}{r:02X}"
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS (H:MM:SS.cc)"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    
    def _group_into_lines(self, words: List[Dict]) -> List[Dict]:
        """Group words into display lines"""
        lines = []
        current_words = []
        
        for word in words:
            current_words.append(word)
            
            if len(current_words) >= self.style.words_per_line:
                lines.append({
                    "words": current_words,
                    "start_time": current_words[0]["start_time"],
                    "end_time": current_words[-1]["end_time"],
                    "text": " ".join(w["word"] for w in current_words)
                })
                current_words = []
        
        # Add remaining
        if current_words:
            lines.append({
                "words": current_words,
                "start_time": current_words[0]["start_time"],
                "end_time": current_words[-1]["end_time"],
                "text": " ".join(w["word"] for w in current_words)
            })
        
        return lines
    
    def _generate_highlight_events(self, line: Dict) -> str:
        """Generate highlight animation events for a line"""
        events = ""
        words = line["words"]
        highlight_color = self._hex_to_ass_color(self.style.highlight_color)
        
        for i, word in enumerate(words):
            # Apply Negative Perceptual Offset
            start_sec = max(0, word["start_time"] + (self.style.npo / 1000.0))
            end_sec = max(0, word["end_time"] + (self.style.npo / 1000.0))
            
            start = self._format_ass_time(start_sec)
            end = self._format_ass_time(end_sec)
            
            # Build text with current word highlighted
            text_parts = []
            for j, w in enumerate(words):
                if j == i:
                    # Highlight current word
                    text_parts.append(f"{{\\c{highlight_color}\\fscx110\\fscy110}}{w['word']}{{\\r}}")
                else:
                    text_parts.append(w['word'])
            
            full_text = " ".join(text_parts)
            events += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{full_text}\n"
        
        return events
    
    def _generate_pop_events(self, line: Dict) -> str:
        """Generate pop-in animation events (one word at a time)"""
        events = ""
        words = line["words"]
        
        for word in words:
            # Apply Negative Perceptual Offset
            start_sec = max(0, word["start_time"] + (self.style.npo / 1000.0))
            end_sec = max(0, word["end_time"] + (self.style.npo / 1000.0))
            
            start = self._format_ass_time(start_sec)
            end = self._format_ass_time(end_sec)
            
            # Pop animation: Elastic Pop (80% -> 110% -> 100%)
            # \t(t1,t2,tags) - animate tags from t1 to t2
            # 0-50ms: Grow to 110%
            # 50-100ms: Settle to 100%
            text = f"{{\\fscx80\\fscy80\\t(0,50,\\fscx110\\fscy110)\\t(50,100,\\fscx100\\fscy100)}}{word['word']}"
            events += f"Dialogue: 0,{start},{end},Highlight,,0,0,0,,{text}\n"
        
        return events
    
    def _generate_wave_events(self, line: Dict) -> str:
        """Generate wave animation events"""
        events = ""
        # Apply Negative Perceptual Offset
        start_sec = max(0, line["start_time"] + (self.style.npo / 1000.0))
        end_sec = max(0, line["end_time"] + (self.style.npo / 1000.0))
        
        start = self._format_ass_time(start_sec)
        end = self._format_ass_time(end_sec)
        
        # Character-by-character wave effect
        text_parts = []
        for i, char in enumerate(line["text"]):
            if char == " ":
                text_parts.append(" ")
            else:
                delay = i * 50  # 50ms delay per character
                text_parts.append(
                    f"{{\\t({delay},{delay+100},\\frz10)\\t({delay+100},{delay+200},\\frz0)}}{char}"
                )
        
        full_text = "".join(text_parts)
        events += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{full_text}\n"
        
        return events
    
    def burn_captions(
        self,
        video_path: str,
        ass_path: str,
        output_path: str,
        use_nvenc: bool = True
    ) -> str:
        """
        Burn captions into video using GPU tensor compositing.
        
        Uses render_to_tensor() to create RGBA overlays for each frame,
        then alpha-composites them onto decoded frames and re-encodes with NVENC.
        
        This avoids FFmpeg's libass dependency which may not be compiled in.
        
        Args:
            video_path: Source video
            ass_path: ASS subtitle file (used to read word data, or ignored if words cached)
            output_path: Output video
            use_nvenc: Use GPU encoding (default True)
            
        Returns:
            Path to output video
        """
        import torch
        from autonomous_trend_agent.editor.gpu_video_utils import (
            decode_video_batched,
            encode_tensor_to_video,
            get_video_info,
            clear_gpu_cache
        )
        
        print("[AnimatedCaptions] Burning captions (GPU compositing)...")
        
        info = get_video_info(video_path)
        fps = info['fps']
        width = info['width']
        height = info['height']
        
        # Get the word data from the line cache (set during generate_ass)
        words = []
        if self._line_cache:
            for line in self._line_cache:
                words.extend(line.get("words", []))
        
        if not words:
            print("[AnimatedCaptions] No word data available, copying source")
            import shutil
            shutil.copy(video_path, output_path)
            return output_path
        
        processed_batches = []
        frame_idx = 0
        
        for batch, batch_info in decode_video_batched(video_path, batch_size=30, device="cuda"):
            # batch shape: (N, C, H, W) float32 [0..1]
            n_frames = batch.shape[0]
            
            for i in range(n_frames):
                timestamp = (frame_idx + i) / fps
                
                # Render caption overlay for this timestamp
                overlay = self.render_to_tensor(words, timestamp, width, height)
                
                if overlay is not None:
                    # overlay shape: (4, H, W) — RGBA float32 [0..1]
                    alpha = overlay[3:4]  # (1, H, W)
                    rgb_overlay = overlay[:3]  # (3, H, W)
                    
                    # Alpha composite: out = fg * alpha + bg * (1 - alpha)
                    frame = batch[i]  # (C, H, W)
                    batch[i] = rgb_overlay * alpha + frame * (1.0 - alpha)
            
            frame_idx += n_frames
            processed_batches.append(batch.cpu())
            clear_gpu_cache()
        
        all_frames = torch.cat(processed_batches, dim=0)
        
        success = encode_tensor_to_video(
            all_frames.to("cuda"),
            output_path,
            fps=fps,
            width=width,
            height=height
        )
        
        if success:
            print("[AnimatedCaptions] Output: %s" % output_path)
        else:
            print("[AnimatedCaptions] Encoding failed!")
        
        clear_gpu_cache()
        return output_path if success else video_path
    
    def add_captions(
        self,
        video_path: str,
        transcript_data: Dict,
        output_path: str,
        use_nvenc: bool = True
    ) -> str:
        """
        Full pipeline: extract words, generate ASS, burn into video.
        
        Args:
            video_path: Source video
            transcript_data: Parakeet/Whisper transcript with word timestamps
            output_path: Output video path
            use_nvenc: Use GPU encoding
            
        Returns:
            Path to captioned video
        """
        # Extract words from transcript
        words = self._extract_words(transcript_data)
        
        if not words:
            print("[AnimatedCaptions] No words found in transcript!")
            import shutil
            shutil.copy(video_path, output_path)
            return output_path
        
        print(f"[AnimatedCaptions] Processing {len(words)} words...")
        
        # Get video dimensions
        width, height = self._get_video_dimensions(video_path)
        
        # Generate ASS file
        ass_path = Path(output_path).with_suffix('.ass')
        self.generate_ass(words, str(ass_path), width, height)
        
        # Burn captions
        result = self.burn_captions(video_path, str(ass_path), output_path, use_nvenc)
        
        # Cleanup ASS file (optional)
        # ass_path.unlink()
        
        return result
    
    def _extract_words(self, transcript_data: Dict) -> List[Dict]:
        """Extract word-level timestamps from transcript"""
        words = []
        
        # Try Parakeet format
        if "result" in transcript_data:
            result = transcript_data["result"]
            if "words" in result:
                for w in result["words"]:
                    words.append({
                        "word": w.get("word", w.get("text", "")),
                        "start_time": w.get("start_time", w.get("start", 0)),
                        "end_time": w.get("end_time", w.get("end", 0))
                    })
        
        # Try Whisper format
        elif "segments" in transcript_data:
            for seg in transcript_data["segments"]:
                if "words" in seg:
                    for w in seg["words"]:
                        words.append({
                            "word": w.get("word", ""),
                            "start_time": w.get("start", 0),
                            "end_time": w.get("end", 0)
                        })
        
        # Try flat words list
        elif "words" in transcript_data:
            for w in transcript_data["words"]:
                words.append({
                    "word": w.get("word", w.get("text", "")),
                    "start_time": w.get("start_time", w.get("start", 0)),
                    "end_time": w.get("end_time", w.get("end", 0))
                })
        
        return words
    
    def _get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        """Get video dimensions using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        data = json.loads(result.stdout)
        
        stream = data.get("streams", [{}])[0]
        return stream.get("width", 1080), stream.get("height", 1920)


def add_animated_captions(
    video_path: str,
    transcript_path: str,
    output_path: str,
    style: str = "tiktok"
) -> str:
    """
    Convenience function to add animated captions.
    
    Args:
        video_path: Source video
        transcript_path: Path to transcript JSON
        output_path: Output video path
        style: Caption style preset
        
    Returns:
        Path to captioned video
    """
    # Load transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    # Create engine and process
    engine = AnimatedCaptionEngine(style=style)
    return engine.add_captions(video_path, transcript, output_path)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*50)
    print("ANIMATED CAPTION ENGINE")
    print("="*50)
    print("\nAvailable styles:")
    for name, style in CAPTION_STYLES.items():
        print(f"  • {name}: {style.font_name}, {style.animation} animation")
    
    if len(sys.argv) >= 4:
        video = sys.argv[1]
        transcript = sys.argv[2]
        output = sys.argv[3]
        style = sys.argv[4] if len(sys.argv) > 4 else "tiktok"
        
        result = add_animated_captions(video, transcript, output, style)
        print(f"\n✅ Output: {result}")
    else:
        print("\nUsage: python animated_captions.py <video> <transcript.json> <output> [style]")

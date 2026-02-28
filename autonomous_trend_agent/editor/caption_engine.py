"""
Auto-Captioning Engine
Generates and burns-in synchronized subtitles using Parakeet transcription data.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict
from datetime import timedelta


class CaptionEngine:
    """
    Generates SRT subtitle files from Parakeet transcription JSON
    and burns them into video using FFmpeg.
    """
    
    def __init__(self, style="tiktok"):
        """
        Args:
            style: Caption style preset ("tiktok", "youtube", "minimal")
        """
        self.style = style
        self.style_configs = {
            "tiktok": {
                "fontsize": 48,
                "fontcolor": "white",
                "bordercolor": "black",
                "borderw": 3,
                "bold": 1,
                "alignment": 10  # Top center
            },
            "youtube": {
                "fontsize": 24,
                "fontcolor": "white",
                "bordercolor": "black",
                "borderw": 2,
                "bold": 0,
                "alignment": 2  # Bottom center
            },
            "minimal": {
                "fontsize": 20,
                "fontcolor": "white",
                "bordercolor": "transparent",
                "borderw": 0,
                "bold": 0,
                "alignment": 2
            }
        }
    
    def add_captions(self, video_path: str, transcription_data: Dict, output_path: str) -> str:
        """
        Generate captions and burn them into video.
        
        Args:
            video_path: Source video path
            transcription_data: Parakeet JSON with word-level timestamps
            output_path: Output video path
            
        Returns:
            Path to captioned video
        """
        print(f"[Captions] Processing: {Path(video_path).name}")
        
        # Extract transcription text
        result = transcription_data.get('result', {})
        text = result.get('text', '')
        words = result.get('words', [])
        
        if not text:
            print("[Captions] WARNING: No transcription text. Skipping captions.")
            # Just copy the source
            import shutil
            shutil.copy(video_path, output_path)
            return output_path
        
        # For v1: Use simple text overlay (no SRT file needed for short clips)
        # Generate subtitle filter string
        subtitle_filter = self._generate_ffmpeg_filter(text)
        
        # Execute FFmpeg with subtitle burn-in
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', subtitle_filter,
            '-c:a', 'copy',  # Copy audio stream
            '-c:v', 'libx264',  # Re-encode video with subtitles
            '-preset', 'fast',
            '-y',  # Overwrite output
            output_path
        ]
        
        print(f"[Captions] Burning subtitles...")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[Captions] Complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[Captions] ERROR: FFmpeg failed")
            print(f"[Captions] STDERR: {e.stderr}")
            raise
    
    def _generate_ffmpeg_filter(self, text: str) -> str:
        """
        Generate FFmpeg drawtext filter for caption overlay.
        
        For v1, we use a simple static overlay.
        Future enhancement: Use SRT file for word-level sync.
        """
        config = self.style_configs[self.style]
        
        # Escape text for FFmpeg
        safe_text = text.replace("'", "\\'").replace(":", "\\:")
        
        # Limit text length for readability
        if len(safe_text) > 100:
            safe_text = safe_text[:97] + "..."
        
        filter_str = (
            f"drawtext="
            f"text='{safe_text}':"
            f"fontsize={config['fontsize']}:"
            f"fontcolor={config['fontcolor']}:"
            f"bordercolor={config['bordercolor']}:"
            f"borderw={config['borderw']}:"
            f"x=(w-text_w)/2:"  # Center horizontally
            f"y=h-th-100"  # 100px from bottom
        )
        
        if config['bold']:
            filter_str += ":font='Arial Bold'"
        
        return filter_str
    
    def generate_srt(self, words: List[Dict], output_path: str) -> str:
        """
        Generate SRT subtitle file from word-level timestamps.
        
        Args:
            words: List of {word, start_time, end_time} dicts
            output_path: Path to save .srt file
            
        Returns:
            Path to SRT file
        """
        # Group words into phrases (2-3 words each)
        phrases = self._group_words_into_phrases(words)
        
        # Write SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, phrase in enumerate(phrases, start=1):
                start_time = self._format_srt_time(phrase['start'])
                end_time = self._format_srt_time(phrase['end'])
                text = phrase['text']
                
                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n")
                f.write("\n")
        
        print(f"[Captions] SRT file: {output_path}")
        return output_path
    
    def _group_words_into_phrases(self, words: List[Dict], max_words=3) -> List[Dict]:
        """Group words into readable phrases."""
        phrases = []
        current_phrase = []
        
        for word in words:
            current_phrase.append(word)
            
            if len(current_phrase) >= max_words:
                phrases.append({
                    'start': current_phrase[0]['start_time'],
                    'end': current_phrase[-1]['end_time'],
                    'text': ' '.join([w['word'] for w in current_phrase])
                })
                current_phrase = []
        
        # Add remaining words
        if current_phrase:
            phrases.append({
                'start': current_phrase[0]['start_time'],
                'end': current_phrase[-1]['end_time'],
                'text': ' '.join([w['word'] for w in current_phrase])
            })
        
        return phrases
    
    def _format_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    engine = CaptionEngine(style="tiktok")
    print("[Captions] Module loaded successfully.")

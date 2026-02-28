# engine/cutter.py
"""
FFmpeg-based Precision Video Cutting Engine
Extracts clips from source video with frame-perfect accuracy
"""
import ffmpeg
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class ClipSpec:
    """Specification for a single clip to extract"""
    start_time: float
    end_time: float
    output_path: str
    metadata: Optional[Dict] = None

class VideoCutter:
    """High-quality video cutting using FFmpeg"""
    
    def __init__(self, quality_preset: str = "medium"):
        """
        Initialize cutter
        
        Args:
            quality_preset: FFmpeg preset (ultrafast, fast, medium, slow, veryslow)
        """
        self.quality_preset = quality_preset
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Verify FFmpeg is installed"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Install from: https://ffmpeg.org/download.html"
            )
    
    def cut_clip(
        self,
        source_video: str,
        start_time: float,
        end_time: float,
        output_path: str,
        audio_only: bool = False
    ) -> Path:
        """
        Extract a single clip with high quality
        
        Args:
            source_video: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds  
            output_path: Where to save the clip
            audio_only: Extract only audio track
            
        Returns:
            Path to output file
        """
        source_video = Path(source_video)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        duration = end_time - start_time
        
        print(f"Cutting clip: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
        
        try:
            if audio_only:
                # Extract audio only
                (
                    ffmpeg
                    .input(str(source_video), ss=start_time, t=duration)
                    .output(str(output_path), acodec='aac', audio_bitrate='192k')
                    .overwrite_output()
                    .run(quiet=True, capture_stderr=True)
                )
            else:
                # Extract video with re-encoding for quality
                (
                    ffmpeg
                    .input(str(source_video), ss=start_time, t=duration)
                    .output(
                        str(output_path),
                        vcodec='libx264',
                        preset=self.quality_preset,
                        crf=18,  # High quality (lower = better, 18 is visually lossless)
                        acodec='aac',
                        audio_bitrate='192k'
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stderr=True)
                )
            
            print(f"Saved to: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg cutting failed: {error_msg}")
    
    def cut_multiple(
        self,
        source_video: str,
        clips: List[ClipSpec],
        progress_callback: Optional[callable] = None
    ) -> List[Path]:
        """
        Extract multiple clips from the same source
        
        Args:
            source_video: Path to source video
            clips: List of clip specifications
            progress_callback: Optional function called after each clip (current, total)
            
        Returns:
            List of paths to generated clips
        """
        output_paths = []
        total = len(clips)
        
        print(f"Extracting {total} clips from {Path(source_video).name}")
        
        for i, clip in enumerate(clips, 1):
            output_path = self.cut_clip(
                source_video,
                clip.start_time,
                clip.end_time,
                clip.output_path
            )
            output_paths.append(output_path)
            
            if progress_callback:
                progress_callback(i, total)
        
        return output_paths
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata (duration, resolution, codec)"""
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            fps_str = video_stream['r_frame_rate']
            try:
                num, den = fps_str.split('/')
                fps_val = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps_val = 30.0

            return {
                "duration": float(probe['format']['duration']),
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": fps_val,
                "codec": video_stream['codec_name']
            }
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to probe video: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python cutter.py <video> <start> <end> <output>")
        print("Example: python cutter.py video.mp4 10.5 45.2 clip.mp4")
        sys.exit(1)
    
    video = sys.argv[1]
    start = float(sys.argv[2])
    end = float(sys.argv[3])
    output = sys.argv[4]
    
    cutter = VideoCutter()
    info = cutter.get_video_info(video)
    print(f"Source: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
    
    cutter.cut_clip(video, start, end, output)
    print("Done!")

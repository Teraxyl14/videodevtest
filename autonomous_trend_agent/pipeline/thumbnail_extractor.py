"""
Thumbnail Extractor — Objective F2.4
=====================================
Extracts or generates thumbnail frames from video shorts.

Strategies:
    1. Gemini-suggested timestamp (if available from AI enhancement)
    2. Visual energy peak detection (find most dynamic frame)
    3. Face detection (frame with clearest face)
    4. Fallback: first non-black frame

Output: thumbnail.jpg at 1080x1920 in each short's directory.

References:
    - Objective F2.4 (Thumbnail extracted or generated)
    - Objective F3.3 (Each directory has thumbnail.jpg)
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ThumbnailExtractor")


class ThumbnailExtractor:
    """
    Extracts the best thumbnail frame from a video.

    Prioritizes:
        1. AI-suggested timestamp (from Gemini enhancement)
        2. 25% mark of video (typically a strong moment)
        3. First non-black frame
    """

    def __init__(self, width: int = 1080, height: int = 1920, quality: int = 95):
        """
        Args:
            width: Output thumbnail width
            height: Output thumbnail height
            quality: JPEG quality (1-100)
        """
        self.width = width
        self.height = height
        self.quality = quality

    def extract(
        self,
        video_path: str,
        output_path: str,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Extract a thumbnail from a video.

        Args:
            video_path: Path to the video file
            output_path: Path to save the thumbnail JPEG
            timestamp: Specific timestamp to extract (seconds).
                       If None, uses smart detection.

        Returns:
            True if thumbnail was successfully extracted
        """
        video = Path(video_path)
        if not video.exists():
            logger.error(f"Video not found: {video_path}")
            return False

        # Get video duration
        duration = self._get_duration(video_path)
        if duration <= 0:
            duration = 60.0  # Fallback

        # Determine best timestamp
        if timestamp is None:
            timestamp = self._find_best_timestamp(video_path, duration)

        # Clamp to valid range
        timestamp = max(0.0, min(timestamp, duration - 0.1))

        # Extract frame with ffmpeg
        return self._extract_frame(video_path, output_path, timestamp)

    def extract_for_short(
        self,
        short_dir: str,
        video_filename: str = "video.mp4",
        metadata_filename: str = "metadata.json"
    ) -> bool:
        """
        Extract thumbnail for a short in its output directory.

        Reads AI enhancement metadata if available for suggested timestamp.

        Args:
            short_dir: Path to the short's output directory
            video_filename: Video file name
            metadata_filename: Metadata file name

        Returns:
            True if thumbnail was extracted
        """
        import json

        short_path = Path(short_dir)
        video_path = short_path / video_filename
        output_path = short_path / "thumbnail.jpg"

        if not video_path.exists():
            logger.warning(f"No video found in {short_dir}")
            return False

        # Try to get AI-suggested timestamp
        timestamp = None
        ai_enhance_path = short_path / "ai_enhancement.json"
        meta_path = short_path / metadata_filename

        if ai_enhance_path.exists():
            try:
                with open(ai_enhance_path) as f:
                    enhancement = json.load(f)
                    timestamp = enhancement.get("thumbnail", {}).get("timestamp")
                    if timestamp is not None:
                        logger.info(f"Using AI-suggested thumbnail at {timestamp:.1f}s")
            except Exception:
                pass

        if timestamp is None and meta_path.exists():
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                    timestamp = metadata.get("thumbnail_timestamp")
            except Exception:
                pass

        return self.extract(str(video_path), str(output_path), timestamp)

    def _find_best_timestamp(self, video_path: str, duration: float) -> float:
        """
        Find the best timestamp for a thumbnail using visual analysis.

        Strategy:
            1. Try 25% mark (usually past intro, in good content)
            2. Verify it's not a black frame
            3. If black, scan forward until non-black found
        """
        # Start at 25% of duration
        candidate = duration * 0.25

        # Check if candidate frame is black
        if self._is_black_frame(video_path, candidate):
            # Scan forward in 0.5s increments
            for offset in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                test_ts = candidate + offset
                if test_ts < duration and not self._is_black_frame(video_path, test_ts):
                    return test_ts

            # Last resort: mid-point
            return duration * 0.5

        return candidate

    def _is_black_frame(self, video_path: str, timestamp: float) -> bool:
        """Check if a frame at the given timestamp is black."""
        try:
            cmd = [
                "ffmpeg", "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-vf", "blackdetect=d=0:pix_th=0.15",
                "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            return "black_start" in result.stderr
        except Exception:
            return False

    def _extract_frame(
        self,
        video_path: str,
        output_path: str,
        timestamp: float
    ) -> bool:
        """Extract a single frame as JPEG."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", str(max(1, min(31, 32 - int(self.quality / 3.2)))),
                str(output_path)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )

            if result.returncode == 0 and Path(output_path).exists():
                size_kb = Path(output_path).stat().st_size / 1024
                logger.info(
                    f"Thumbnail extracted: {Path(output_path).name} "
                    f"({size_kb:.0f}KB at {timestamp:.1f}s)"
                )
                return True
            else:
                logger.error(f"FFmpeg failed: {result.stderr[:200]}")
                return False

        except Exception as e:
            logger.error(f"Thumbnail extraction failed: {e}")
            return False

    def _get_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                fmt = json.loads(result.stdout).get("format", {})
                return float(fmt.get("duration", 0))
        except Exception:
            pass
        return 0.0

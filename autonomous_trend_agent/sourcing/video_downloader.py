"""
Video Downloader (Phase 2, C2.1 + C2.2)

Downloads a selected YouTube video using yt-dlp with full metadata,
best available quality up to 1080p, and validates the output file exists.

Objectives:
    C2.1: Video downloaded to downloaded_videos/ directory
    C2.2: Download includes metadata (JSON sidecar file)
    C2.3: Partial/corrupted downloads detected and cleaned up (H3.2/H3.3)
"""

import os
import json
import shutil
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a yt-dlp download operation."""
    success: bool
    video_path: str             # Absolute path to downloaded .mp4
    metadata_path: str          # Absolute path to .info.json sidecar
    title: str
    channel: str
    duration_sec: int
    file_size_bytes: int
    error: Optional[str] = None


class VideoDownloader:
    """
    Downloads YouTube videos using yt-dlp at best quality (max 1080p).

    Prefers:  bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]
    Output:   {download_dir}/{sanitized_title}/video.mp4
    Metadata: {download_dir}/{sanitized_title}/metadata.json  (C2.2)

    Falls back gracefully if yt-dlp is not on PATH (Windows host),
    but works fully inside Docker.
    """

    FORMAT = (
        "bestvideo[height<=1080][vcodec!^=av01][ext=mp4]+bestaudio[ext=m4a]"
        "/bestvideo[height<=1080][vcodec!^=av01]+bestaudio"
        "/best[height<=1080][vcodec!^=av01]"
        "/best"
    )

    def __init__(
        self,
        download_dir: Optional[str] = None,
        ytdlp_path: str = "yt-dlp",
    ):
        self.download_dir = Path(
            download_dir
            or os.getenv("DOWNLOAD_DIR", "/app/downloaded_videos")
        )
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.ytdlp = ytdlp_path
        self._check_ytdlp()
        logger.info(f"[VideoDownloader] Downloads → {self.download_dir}")

    def download(self, video_url: str, title_hint: str = "") -> DownloadResult:
        """
        Download a video from YouTube.

        Args:
            video_url:   Full YouTube URL (e.g. https://youtube.com/watch?v=...)
            title_hint:  Used for output directory naming if yt-dlp title unavailable.

        Returns:
            DownloadResult with paths to video.mp4 and metadata.json.
        """
        # Create a clean output subdirectory
        safe_title = _sanitize(title_hint or "video")[:60]
        out_dir = self.download_dir / safe_title
        out_dir.mkdir(parents=True, exist_ok=True)

        video_out = out_dir / "video.mp4"
        info_json = out_dir / "metadata.json"

        logger.info(f"[VideoDownloader] Starting download: {video_url}")
        logger.info(f"[VideoDownloader] Output dir: {out_dir}")

        # yt-dlp command
        cmd = [
            self.ytdlp,

            # Format selection
            "--format", self.FORMAT,
            "--merge-output-format", "mp4",

            # Output path (fixed filename inside the subdirectory)
            "--output", str(video_out),

            # Metadata sidecar (C2.2)
            "--write-info-json",
            "--write-thumbnail",

            # Performance
            "--no-playlist",
            "--concurrent-fragments", "4",

            # Enforce max duration at download time (defends against sourcing fallback)
            "--match-filter", "duration <= 1800",  # ~30 minutes max (strict per C3.3)

            # Retry logic (H3.1)
            "--retries", "3",
            "--fragment-retries", "3",
            "--continue",           # Resume partial downloads

            # Embed metadata into MP4 container
            "--embed-metadata",
            "--embed-chapters",

            "--no-warnings",
            "--progress",

            video_url,
        ]
        
        # Replace binary call with module call for Docker compatibility
        if cmd[0] == 'yt-dlp':
            import sys
            cmd = [sys.executable, "-m", "yt_dlp"] + cmd[1:]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,    # Capture stderr for error diagnosis
                text=True,
                timeout=3600,           # 1 hr max for a 60-min video
                check=True,
                encoding='utf-8',
                errors='replace'
            )
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, 'stderr', '') or ''
            # Extract actionable error from yt-dlp output
            err_detail = ''
            for line in stderr.split('\n'):
                line_lower = line.lower()
                if any(k in line_lower for k in [
                    'error', 'unavailable', 'private', 'age', 'copyright',
                    'blocked', 'removed', 'terminated', 'sign in',
                ]):
                    err_detail = line.strip()
                    break
            err = f"yt-dlp exited with code {e.returncode}"
            if err_detail:
                err += f": {err_detail}"
            logger.error(f"[VideoDownloader] {err}")
            self._cleanup_partial(out_dir, video_out)
            return DownloadResult(
                success=False, video_path="", metadata_path="",
                title=title_hint, channel="", duration_sec=0,
                file_size_bytes=0, error=err,
            )
        except subprocess.TimeoutExpired:
            err = "Download timed out after 1 hour"
            logger.error(f"[VideoDownloader] {err}")
            self._cleanup_partial(out_dir, video_out)
            return DownloadResult(
                success=False, video_path="", metadata_path="",
                title=title_hint, channel="", duration_sec=0,
                file_size_bytes=0, error=err,
            )
        except FileNotFoundError:
            err = f"yt-dlp not found at '{self.ytdlp}'"
            logger.error(f"[VideoDownloader] {err}")
            return DownloadResult(
                success=False, video_path="", metadata_path="",
                title=title_hint, channel="", duration_sec=0,
                file_size_bytes=0, error=err,
            )

        # Verify output file exists
        if not video_out.exists():
            # yt-dlp sometimes adjusts the extension to .mkv or .webm — scan the folder
            other_videos = list(out_dir.glob("video.*")) + list(out_dir.glob("*.mkv")) + list(out_dir.glob("*.webm"))
            actual_video = None
            for p in other_videos:
                if p.suffix.lower() in [".mkv", ".webm", ".mp4", ".mov", ".avi"]:
                    actual_video = p
                    break
            
            if actual_video:
                if actual_video.name != "video.mp4":
                    logger.info(f"[VideoDownloader] Remuxing {actual_video.name} to video.mp4 via FFmpeg")
                    remux_cmd = [
                        "ffmpeg", "-y", "-i", str(actual_video),
                        "-c", "copy", str(video_out)
                    ]
                    try:
                        subprocess.run(remux_cmd, capture_output=True, check=True)
                        actual_video.unlink(missing_ok=True)  # Clean up the .mkv/.webm
                    except subprocess.CalledProcessError as e:
                        err = f"Failed to remux {actual_video.name} to mp4"
                        logger.error(f"[VideoDownloader] {err}")
                        return DownloadResult(
                            success=False, video_path="", metadata_path="",
                            title=title_hint, channel="", duration_sec=0,
                            file_size_bytes=0, error=err,
                        )
            else:
                err = "Download completed but no video file found in output directory"
                logger.error(f"[VideoDownloader] {err}")
                return DownloadResult(
                    success=False, video_path="", metadata_path="",
                    title=title_hint, channel="", duration_sec=0,
                    file_size_bytes=0, error=err,
                )

        # Minimum file size check — reject corrupt/truncated downloads
        MIN_SIZE_BYTES = 1_000_000  # 1MB minimum
        actual_size = video_out.stat().st_size
        if actual_size < MIN_SIZE_BYTES:
            err = f"Downloaded file too small ({actual_size / 1024:.0f} KB) — likely corrupt"
            logger.error(f"[VideoDownloader] {err}")
            self._cleanup_partial(out_dir, video_out)
            return DownloadResult(
                success=False, video_path="", metadata_path="",
                title=title_hint, channel="", duration_sec=0,
                file_size_bytes=actual_size, error=err,
            )

        # Read metadata from info.json sidecar
        title, channel, duration = title_hint, "", 0
        info_jsons = list(out_dir.glob("*.info.json"))
        if info_jsons:
            raw_info = info_jsons[0]
            try:
                with open(raw_info, encoding="utf-8") as f:
                    info = json.load(f)
                title = info.get("title", title_hint)
                channel = info.get("uploader", "")
                duration = int(info.get("duration", 0))
                # Save clean metadata.json (C2.2)
                clean_meta = {
                    "title": title,
                    "channel": channel,
                    "uploader_id": info.get("uploader_id", ""),
                    "url": info.get("webpage_url", video_url),
                    "duration_sec": duration,
                    "view_count": info.get("view_count", 0),
                    "like_count": info.get("like_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "description": info.get("description", "")[:1000],
                    "tags": info.get("tags", [])[:10],
                    "categories": info.get("categories", []),
                    "resolution": f"{info.get('width', 0)}x{info.get('height', 0)}",
                    "fps": info.get("fps", 0),
                    "filesize_bytes": video_out.stat().st_size,
                }
                with open(info_json, "w", encoding="utf-8") as f:
                    json.dump(clean_meta, f, indent=2)
            except Exception as e:
                logger.warning(f"[VideoDownloader] Metadata parse failed: {e}")

        file_size = video_out.stat().st_size
        logger.info(
            f"[VideoDownloader] ✅ Download complete: {video_out.name} "
            f"({file_size / 1e6:.1f} MB)"
        )

        return DownloadResult(
            success=True,
            video_path=str(video_out),
            metadata_path=str(info_json) if info_json.exists() else "",
            title=title,
            channel=channel,
            duration_sec=duration,
            file_size_bytes=file_size,
        )

    def _cleanup_partial(self, out_dir: Path, video_file: Path):
        """Remove partial download artifacts (H3.3 — no corrupted outputs)."""
        try:
            if video_file.exists():
                video_file.unlink()
                logger.info(f"[VideoDownloader] Cleaned partial: {video_file.name}")
            # Also clean .part files
            for part in out_dir.glob("*.part"):
                part.unlink()
        except Exception as e:
            logger.warning(f"[VideoDownloader] Cleanup error: {e}")

    def _check_ytdlp(self):
        """Non-fatal check — yt-dlp only needed inside Docker."""
        if not shutil.which(self.ytdlp):
            logger.warning(
                f"[VideoDownloader] '{self.ytdlp}' not found on PATH. "
                "Downloads will fail outside Docker. This is expected on Windows host."
            )


def _sanitize(name: str) -> str:
    """Remove filesystem-unsafe characters from directory name."""
    import re
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" ._")

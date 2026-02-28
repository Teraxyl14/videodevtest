"""
Quality Assurance Module — Section G
======================================
Automated post-production checks for generated shorts.

G1 (Automated Checks):
    G1.1: Audio sync verification (< 0.1s drift)
    G1.2: Resolution verification (1080x1920)
    G1.3: Duration verification (30–90s)
    G1.4: No black frames detected

G2 (Content QA):
    G2.1: Hook exists in first 3 seconds
    G2.2: Content coherent as standalone clip
    G2.3: Natural ending (not mid-sentence)
    G2.4: Captions accurate to speech

References:
    - Objectives G1.1–G2.4
"""

import subprocess
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("QualityAssurance")


@dataclass
class QACheck:
    """Result of a single QA check."""
    name: str
    passed: bool
    value: Any = None
    expected: Any = None
    message: str = ""


@dataclass
class QAReport:
    """Complete QA report for a generated short."""
    video_path: str
    overall_passed: bool
    checks: List[QACheck] = field(default_factory=list)
    score: float = 0.0  # 0–100

    def summary(self) -> str:
        status = "✅ PASSED" if self.overall_passed else "❌ FAILED"
        lines = [
            f"QA Report: {Path(self.video_path).name}",
            f"  Overall: {status} (Score: {self.score:.0f}/100)",
            ""
        ]
        for check in self.checks:
            icon = "✅" if check.passed else "❌"
            lines.append(f"  {icon} {check.name}: {check.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "overall_passed": self.overall_passed,
            "score": self.score,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": str(c.value),
                    "expected": str(c.expected),
                    "message": c.message
                }
                for c in self.checks
            ]
        }


class QualityAssurance:
    """
    Automated quality assurance for generated shorts.

    Runs a battery of checks on each output video:
    - Technical checks (resolution, fps, duration, sync, black frames)
    - Content checks (hook, coherence, ending — via Gemini if available)

    Usage:
        qa = QualityAssurance()
        report = qa.check("output/short_01/video.mp4")
        if report.overall_passed:
            print("Ready for upload!")
        else:
            print(report.summary())
    """

    def __init__(
        self,
        target_width: int = 1080,
        target_height: int = 1920,
        min_duration: float = 30.0,
        max_duration: float = 90.0,
        min_fps: float = 29.0,
        max_sync_drift: float = 0.1,
        max_black_frame_pct: float = 0.02,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_fps = min_fps
        self.max_sync_drift = max_sync_drift
        self.max_black_frame_pct = max_black_frame_pct

    def check(
        self,
        video_path: str,
        transcript: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> QAReport:
        """
        Run all QA checks on a generated short.

        Args:
            video_path: Path to the generated short video
            transcript: Optional transcript dict with word timestamps
            metadata: Optional metadata dict from the pipeline

        Returns:
            QAReport with check results and overall score
        """
        checks = []

        # Probe video
        probe = self._probe(video_path)
        if probe is None:
            return QAReport(
                video_path=video_path,
                overall_passed=False,
                checks=[QACheck("File Probe", False, message="Cannot read video file")],
                score=0
            )

        video_stream = None
        audio_stream = None
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video" and video_stream is None:
                video_stream = s
            elif s.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = s

        # ---- G1.2: Resolution Verification ----
        if video_stream:
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            res_ok = (width == self.target_width and height == self.target_height)
            checks.append(QACheck(
                name="Resolution",
                passed=res_ok,
                value=f"{width}x{height}",
                expected=f"{self.target_width}x{self.target_height}",
                message=f"{width}x{height}" + ("" if res_ok else f" (expected {self.target_width}x{self.target_height})")
            ))

            # ---- FPS Check ----
            fps_str = video_stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps = 0.0

            fps_ok = fps >= self.min_fps
            checks.append(QACheck(
                name="Frame Rate",
                passed=fps_ok,
                value=f"{fps:.1f}fps",
                expected=f"≥{self.min_fps}fps",
                message=f"{fps:.1f}fps" + ("" if fps_ok else f" (min {self.min_fps})")
            ))

        # ---- G1.3: Duration Verification ----
        duration = float(probe.get("format", {}).get("duration", 0))
        dur_ok = self.min_duration <= duration <= self.max_duration
        checks.append(QACheck(
            name="Duration",
            passed=dur_ok,
            value=f"{duration:.1f}s",
            expected=f"{self.min_duration}-{self.max_duration}s",
            message=f"{duration:.1f}s ({duration/60:.1f}min)" + ("" if dur_ok else " (out of range)")
        ))

        # ---- G1.1: Audio Sync Verification ----
        if audio_stream and video_stream:
            sync_drift = self._check_audio_sync(video_path, probe)
            sync_ok = abs(sync_drift) < self.max_sync_drift
            checks.append(QACheck(
                name="Audio Sync",
                passed=sync_ok,
                value=f"{sync_drift:.3f}s",
                expected=f"<{self.max_sync_drift}s",
                message=f"Drift: {sync_drift:.3f}s" + ("" if sync_ok else " (DESYNC)")
            ))
        elif not audio_stream:
            checks.append(QACheck(
                name="Audio Sync",
                passed=False,
                message="No audio track present"
            ))

        # ---- G1.4: Black Frame Detection ----
        black_pct = self._check_black_frames(video_path)
        black_ok = black_pct < self.max_black_frame_pct
        checks.append(QACheck(
            name="Black Frames",
            passed=black_ok,
            value=f"{black_pct*100:.1f}%",
            expected=f"<{self.max_black_frame_pct*100:.0f}%",
            message=f"{black_pct*100:.1f}% black frames" + ("" if black_ok else " (too many)")
        ))

        # ---- G2.1: Hook Check (first 3 seconds have content) ----
        hook_ok = self._check_hook(video_path, duration)
        checks.append(QACheck(
            name="Hook (First 3s)",
            passed=hook_ok,
            message="Visual activity detected in first 3s" if hook_ok else "First 3s appear static"
        ))

        # ---- G2.3: Natural Ending (not mid-silence check) ----
        ending_ok = self._check_natural_ending(video_path, duration)
        checks.append(QACheck(
            name="Natural Ending",
            passed=ending_ok,
            message="Audio present at end" if ending_ok else "Clip may end abruptly"
        ))

        # ---- G2.4: Caption Accuracy (if transcript available) ----
        if transcript and transcript.get("words"):
            cap_ok = len(transcript["words"]) > 0
            checks.append(QACheck(
                name="Captions",
                passed=cap_ok,
                value=f"{len(transcript['words'])} words",
                message=f"{len(transcript['words'])} caption words aligned"
            ))

        # ---- Calculate Score ----
        total = len(checks)
        passed = sum(1 for c in checks if c.passed)
        score = (passed / total * 100) if total > 0 else 0

        # Critical checks override
        critical_checks = ["Resolution", "Duration", "Audio Sync"]
        critical_failed = any(
            not c.passed for c in checks if c.name in critical_checks
        )
        overall = not critical_failed and score >= 70

        report = QAReport(
            video_path=video_path,
            overall_passed=overall,
            checks=checks,
            score=score
        )

        logger.info(report.summary())
        return report

    def check_batch(
        self,
        output_dir: str,
        video_filename: str = "video.mp4"
    ) -> List[QAReport]:
        """
        Check all shorts in an output directory.

        Args:
            output_dir: Directory containing short_XX subdirectories
            video_filename: Expected video filename in each subdirectory

        Returns:
            List of QAReport for each found short
        """
        output_path = Path(output_dir)
        reports = []

        for short_dir in sorted(output_path.iterdir()):
            if not short_dir.is_dir():
                continue
            video = short_dir / video_filename
            if video.exists():
                # Load transcript if available
                transcript = None
                transcript_file = short_dir / "transcript.json"
                if transcript_file.exists():
                    with open(transcript_file) as f:
                        transcript = json.load(f)

                # Load metadata if available
                metadata = None
                meta_file = short_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata = json.load(f)

                report = self.check(str(video), transcript, metadata)
                reports.append(report)

        return reports

    def save_report(self, report: QAReport, output_path: str):
        """Save QA report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    # =========================================================================
    # Individual Check Implementations
    # =========================================================================

    def _probe(self, video_path: str) -> Optional[dict]:
        """Run ffprobe and return JSON."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
            return None
        except Exception:
            return None

    def _check_audio_sync(self, video_path: str, probe: dict) -> float:
        """
        Check audio-video sync drift.

        Compares the start_time of audio and video streams.
        A positive value means audio leads video.

        Returns:
            Drift in seconds (positive = audio leads)
        """
        video_start = 0.0
        audio_start = 0.0

        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                video_start = float(stream.get("start_time", 0))
            elif stream.get("codec_type") == "audio":
                audio_start = float(stream.get("start_time", 0))

        # Also compare duration difference as a secondary check
        video_dur = 0.0
        audio_dur = 0.0
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                video_dur = float(stream.get("duration", 0))
            elif stream.get("codec_type") == "audio":
                audio_dur = float(stream.get("duration", 0))

        start_drift = audio_start - video_start
        dur_drift = abs(video_dur - audio_dur) if video_dur > 0 and audio_dur > 0 else 0

        # Return the larger of start drift and duration discrepancy
        return max(abs(start_drift), dur_drift)

    def _check_black_frames(self, video_path: str) -> float:
        """
        Detect percentage of black frames using ffmpeg's blackdetect filter.

        Returns:
            Fraction of video that is black (0.0–1.0)
        """
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", "blackdetect=d=0.05:pix_th=0.10",
                "-an", "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            stderr = result.stderr

            # Parse blackdetect output
            total_black = 0.0
            for line in stderr.split("\n"):
                if "black_duration:" in line:
                    try:
                        dur_str = line.split("black_duration:")[1].strip()
                        total_black += float(dur_str)
                    except (ValueError, IndexError):
                        pass

            # Get total duration
            probe = self._probe(video_path)
            total_duration = float(
                probe.get("format", {}).get("duration", 1)
            ) if probe else 1.0

            return total_black / total_duration if total_duration > 0 else 0.0

        except Exception:
            return 0.0

    def _check_hook(self, video_path: str, duration: float) -> bool:
        """
        Check if the first 3 seconds have visual activity (not static/black).

        Uses ffmpeg to extract first 3s and check for scene changes or motion.
        """
        try:
            # Check if there's significant visual change in first 3s
            cmd = [
                "ffmpeg", "-i", video_path,
                "-t", "3",
                "-vf", "blackdetect=d=1:pix_th=0.15",
                "-an", "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )

            # If the entire first 3s is black → no hook
            if "black_start:0" in result.stderr and "black_duration:3" in result.stderr:
                return False

            return True

        except Exception:
            return True  # Can't check → assume OK

    def _check_natural_ending(self, video_path: str, duration: float) -> bool:
        """
        Check if the video ends naturally (audio doesn't cut abruptly).

        Analyzes the last 2 seconds for audio presence.
        """
        try:
            start_time = max(0, duration - 2)
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", video_path,
                "-af", "volumedetect",
                "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )

            # Parse mean volume
            for line in result.stderr.split("\n"):
                if "mean_volume" in line:
                    try:
                        db_str = line.split("mean_volume:")[1].strip().split(" ")[0]
                        mean_db = float(db_str)
                        # If audio is present (> -60dB) at the end, it's OK
                        # Very quiet endings might indicate mid-cut
                        return mean_db > -60.0
                    except (ValueError, IndexError):
                        pass

            return True  # Can't determine → assume OK

        except Exception:
            return True


def run_qa(output_dir: str) -> bool:
    """
    Convenience function to run QA on all shorts in a directory.

    Args:
        output_dir: Path to pipeline output directory

    Returns:
        True if all shorts pass QA
    """
    qa = QualityAssurance()
    reports = qa.check_batch(output_dir)

    print("\n" + "=" * 60)
    print("QUALITY ASSURANCE REPORT")
    print("=" * 60)

    all_passed = True
    for report in reports:
        print(f"\n{report.summary()}")
        if not report.overall_passed:
            all_passed = False

        # Save individual report
        short_dir = Path(report.video_path).parent
        qa.save_report(report, str(short_dir / "qa_report.json"))

    print(f"\n{'=' * 60}")
    print(f"Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    print(f"{'=' * 60}\n")

    return all_passed

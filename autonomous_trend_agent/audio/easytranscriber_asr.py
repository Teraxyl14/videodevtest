"""
easytranscriber ASR Module (Phase 3, D1.1-D1.4)

Wraps the easytranscriber pipeline() function from KBLab for
GPU-accelerated transcription with word-level timestamps.

Replaces: whisperx_transcriber.py, parakeet_transcriber.py

Key differences from WhisperX:
- Uses pipeline() function (not a class)
- GPU-parallelized forced alignment via PyTorch's Viterbi API
- 35-102% faster than WhisperX
- Depends on easyaligner for forced alignment

Objectives: D1.1 (transcription), D1.2 (word-level timestamps),
            D1.3 (<10% WER), D1.4 (>10x realtime GPU speed)
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """A single word with its timestamp and confidence."""
    text: str
    start: float           # seconds
    end: float             # seconds
    score: float = 1.0     # alignment confidence


@dataclass
class TranscriptSegment:
    """A sentence-level segment with word timestamps."""
    text: str
    start: float
    end: float
    score: float = 1.0
    words: List[WordTimestamp] = field(default_factory=list)
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription output for a single audio file."""
    segments: List[TranscriptSegment]
    language: str = "en"
    duration: float = 0.0
    model: str = "distil-whisper/distil-large-v3.5"

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "score": s.score,
                    "speaker": s.speaker,
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end, "score": w.score}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
        }

    def save(self, path: str):
        """Save transcription to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class EasyTranscriberConfig:
    """Configuration for the easytranscriber pipeline."""
    vad_model: str = "pyannote"
    emissions_model: str = "facebook/wav2vec2-base-960h"
    transcription_model: str = "distil-whisper/distil-large-v3.5"
    language: str = "en"
    device: str = "cuda"
    output_dir: str = "/app/output/transcriptions"
    batch_size: int = 16


class EasyTranscriberASR:
    """
    GPU-accelerated transcription wrapper around easytranscriber's
    pipeline() function.

    Always resident in VRAM as part of the Parallel Residency model
    (~2GB footprint with distil-large-v3.5 + wav2vec2-base-960h).

    Usage:
        asr = EasyTranscriberASR()
        result = asr.transcribe("/path/to/audio.wav")
        print(result.segments[0].words)
    """

    def __init__(self, config: Optional[EasyTranscriberConfig] = None):
        self.config = config or EasyTranscriberConfig()
        self._pipeline_fn = None
        self._tokenizer = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-load the pipeline and tokenizer on first use."""
        if self._initialized:
            return

        try:
            from easytranscriber.pipelines import pipeline
            from easyaligner.text import load_tokenizer

            self._pipeline_fn = pipeline
            self._tokenizer = load_tokenizer(self.config.language)
            self._initialized = True
            logger.info(
                f"[EasyTranscriberASR] Initialized "
                f"(model={self.config.transcription_model}, "
                f"device={self.config.device})"
            )
        except ImportError as e:
            logger.error(
                f"[EasyTranscriberASR] Import failed: {e}. "
                f"Install with: pip install easytranscriber"
            )
            raise

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe a single audio/video file with word-level timestamps.

        Args:
            audio_path: Path to audio file (.wav, .mp3, .flac) or video (.mp4, .mkv)

        Returns:
            TranscriptionResult with segments and word timestamps.
        """
        self._ensure_initialized()
        audio_path = str(audio_path)
        audio_dir = str(Path(audio_path).parent)
        audio_name = Path(audio_path).name

        # Create output directory for this run
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[EasyTranscriberASR] Transcribing: {audio_name}")

        try:
            # Run the easytranscriber pipeline
            self._pipeline_fn(
                vad_model=self.config.vad_model,
                emissions_model=self.config.emissions_model,
                transcription_model=self.config.transcription_model,
                audio_paths=[audio_name],
                audio_dir=audio_dir,
                output_dir=str(output_dir),
            )

            # Parse the output alignment JSON
            result = self._parse_output(audio_name, output_dir)
            logger.info(
                f"[EasyTranscriberASR] Done: {len(result.segments)} segments, "
                f"{result.word_count} words, {result.duration:.1f}s"
            )
            return result

        except Exception as e:
            logger.error(f"[EasyTranscriberASR] Transcription failed: {e}")
            raise

    def transcribe_batch(self, audio_paths: List[str]) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in a single pipeline run.

        Leverages easytranscriber's parallel loading and batched emission
        extraction for maximum throughput.
        """
        self._ensure_initialized()

        if not audio_paths:
            return []

        # All files must be in the same directory for pipeline()
        audio_dir = str(Path(audio_paths[0]).parent)
        audio_names = [Path(p).name for p in audio_paths]

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[EasyTranscriberASR] Batch transcribing {len(audio_names)} files...")

        try:
            self._pipeline_fn(
                vad_model=self.config.vad_model,
                emissions_model=self.config.emissions_model,
                transcription_model=self.config.transcription_model,
                audio_paths=audio_names,
                audio_dir=audio_dir,
                output_dir=str(output_dir),
            )

            results = []
            for name in audio_names:
                result = self._parse_output(name, output_dir)
                results.append(result)

            logger.info(f"[EasyTranscriberASR] Batch done: {len(results)} files processed.")
            return results

        except Exception as e:
            logger.error(f"[EasyTranscriberASR] Batch transcription failed: {e}")
            raise

    def _parse_output(
        self, audio_name: str, output_dir: Path
    ) -> TranscriptionResult:
        """
        Parse the easytranscriber output JSON into our TranscriptionResult format.

        easytranscriber outputs:
            output_dir/alignments/<audio_name>.json
        """
        # Look for alignment output
        stem = Path(audio_name).stem
        alignment_dir = output_dir / "alignments"
        alignment_file = alignment_dir / f"{stem}.json"

        if not alignment_file.exists():
            # Try alternate locations
            for candidate in [
                output_dir / f"{stem}.json",
                output_dir / "output" / "alignments" / f"{stem}.json",
            ]:
                if candidate.exists():
                    alignment_file = candidate
                    break

        if not alignment_file.exists():
            logger.warning(
                f"[EasyTranscriberASR] No alignment output found for {audio_name}. "
                f"Looked in: {alignment_dir}"
            )
            return TranscriptionResult(segments=[], language=self.config.language)

        with open(alignment_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            raw_segments = data.get("segments", data.get("alignments", [data]))
        elif isinstance(data, list):
            raw_segments = data
        else:
            raw_segments = []

        segments: List[TranscriptSegment] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue

            words = []
            for w in seg.get("words", []):
                words.append(WordTimestamp(
                    text=w.get("text", "").strip(),
                    start=float(w.get("start", 0)),
                    end=float(w.get("end", 0)),
                    score=float(w.get("score", 1.0)),
                ))

            segments.append(TranscriptSegment(
                text=seg.get("text", "").strip(),
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                score=float(seg.get("score", 1.0)),
                words=words,
                speaker=seg.get("speaker"),
            ))

        # Compute duration from last segment
        duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            segments=segments,
            language=self.config.language,
            duration=duration,
            model=self.config.transcription_model,
        )

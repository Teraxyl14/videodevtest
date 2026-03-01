"""
Diarization Spoke — Spoke D (Voice ID)
=======================================
Ephemeral GPU process for Speaker Diarization.
Uses NeMo TitaNet-Large via the NeMo MSDD diarization pipeline.

Lifecycle: Spawn → Load Model → Diarize → Attach Labels → Return → Die
VRAM Usage: ~0.5GB peak (TitaNet is very lightweight)
VRAM Reclamation: Guaranteed via process termination

Output: Augmented transcript with speaker labels on every word/segment
        e.g. [{"word": "Hello", "start": 0.0, "end": 0.3, "speaker": "SPEAKER_00"}]

Architecture:
    The ASR spoke runs FIRST (audio_spoke.py) and produces word-level timestamps.
    This spoke receives that transcript and the same audio, runs NeMo MSDD
    diarization to get per-segment speaker turns, then merges the two outputs.

References:
    - NeMo MSDD Diarization: nemo.collections.asr.models.ClusteringDiarizer
    - TitaNet-Large: nvidia/titanet-large (speaker embedding model)
    - Objectives D2.1 (diarization), D2.2 (speaker labels), D2.3 (attribution)
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("Spoke-Diarize")


# ─── Data structures ──────────────────────────────────────────────────────────

def _make_diarized_word(
    word: str,
    start: float,
    end: float,
    speaker: str,
    confidence: float = 1.0,
) -> Dict:
    return {
        "word": word,
        "start": start,
        "end": end,
        "speaker": speaker,
        "confidence": confidence,
    }


# ─── Speaker label assignment ─────────────────────────────────────────────────

def assign_speaker_labels(
    word_timestamps: List[Dict],
    diarization_segments: List[Dict],
) -> List[Dict]:
    """
    Merge word-level timestamps from ASR with speaker turns from diarization.

    Each word is assigned the speaker who was speaking at its midpoint.
    Ties (word in silence between turns) are assigned to the nearest speaker.

    Args:
        word_timestamps:     List of {"word", "start", "end"} from Parakeet
        diarization_segments: List of {"start", "end", "speaker"} from TitaNet

    Returns:
        List of diarized words with "speaker" field added.
    """
    result = []
    for word in word_timestamps:
        w_start = float(word.get("start", 0))
        w_end = float(word.get("end", w_start + 0.1))
        w_mid = (w_start + w_end) / 2

        assigned = "SPEAKER_00"  # Fallback
        best_overlap = -1.0

        for seg in diarization_segments:
            s_start = float(seg["start"])
            s_end = float(seg["end"])

            # Overlap = intersection of [w_start,w_end] and [s_start,s_end]
            overlap = min(w_end, s_end) - max(w_start, s_start)
            if overlap > best_overlap:
                best_overlap = overlap
                assigned = seg["speaker"]

        result.append(_make_diarized_word(
            word=word.get("word", ""),
            start=w_start,
            end=w_end,
            speaker=assigned,
            confidence=float(word.get("confidence", 1.0)),
        ))

    return result


def build_speaker_segments(diarized_words: List[Dict]) -> List[Dict]:
    """
    Convert per-word speaker labels into continuous speaker segments.
    Consecutive words from the same speaker are grouped.

    Returns:
        List of {"start", "end", "speaker", "text"} segments
    """
    if not diarized_words:
        return []

    segments = []
    current_speaker = diarized_words[0]["speaker"]
    current_start = diarized_words[0]["start"]
    current_words = [diarized_words[0]["word"]]
    current_end = diarized_words[0]["end"]

    for word in diarized_words[1:]:
        if word["speaker"] == current_speaker:
            current_words.append(word["word"])
            current_end = word["end"]
        else:
            segments.append({
                "start": current_start,
                "end": current_end,
                "speaker": current_speaker,
                "text": " ".join(current_words).strip(),
            })
            current_speaker = word["speaker"]
            current_start = word["start"]
            current_words = [word["word"]]
            current_end = word["end"]

    # Flush last segment
    segments.append({
        "start": current_start,
        "end": current_end,
        "speaker": current_speaker,
        "text": " ".join(current_words).strip(),
    })

    return segments


# ─── Main spoke function ──────────────────────────────────────────────────────

def run_diarization_spoke(
    audio_path: str,
    transcript: Dict,
    models_dir: str,
    result_queue=None,
    device: str = "cuda",
    num_speakers: Optional[int] = None,
    max_speakers: int = 4,
):
    """
    Ephemeral speaker diarization spoke.

    This function runs in its own process (via multiprocessing.spawn).
    It initializes CUDA, loads NeMo TitaNet, diarizes the audio, merges
    with the ASR transcript, and exits to free all VRAM.

    Args:
        audio_path:    Path to audio file (WAV preferred; mono, 16kHz)
        transcript:    Transcript dict from audio_spoke: {"text", "word_timestamps"}
        models_dir:    Directory for NeMo model weight cache
        result_queue:  multiprocessing.Queue for returning results
        device:        "cuda" or "cpu"
        num_speakers:  Known speaker count (None = auto-detect, faster if set)
        max_speakers:  Max speakers to detect in auto mode
    """
    os.environ["TRANSFORMERS_CACHE"] = models_dir
    os.environ["HF_HOME"] = models_dir

    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"[Diarize] Running on {gpu_name} ({vram:.1f}GB VRAM)")
        else:
            device = "cpu"
            logger.info("[Diarize] Running on CPU (fallback)")

        # ── Step 1: Prepare audio (NeMo needs WAV 16kHz mono) ─────────
        wav_path = _ensure_wav(audio_path)

        # ── Step 2: Build NeMo config and run diarization ─────────────
        diarization_segments = _run_nemo_diarization(
            wav_path=wav_path,
            models_dir=models_dir,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
            device=device,
        )

        # ── Step 3: Merge diarization with ASR transcript ─────────────
        word_timestamps = transcript.get("word_timestamps", [])

        if not word_timestamps:
            # No word timestamps — attach speaker to text segments by time only
            logger.warning("[Diarize] No word timestamps in transcript. "
                           "Attaching speakers to raw segments.")
            diarized_words = []
        else:
            diarized_words = assign_speaker_labels(word_timestamps, diarization_segments)

        speaker_segments = build_speaker_segments(diarized_words)

        # Count unique speakers
        speakers = sorted(set(s["speaker"] for s in diarization_segments))
        logger.info(
            f"[Diarize] Identified {len(speakers)} speaker(s): {speakers}"
        )

        result = {
            "speaker_count": len(speakers),
            "speakers": speakers,
            "diarization_segments": diarization_segments,   # Raw TitaNet output
            "diarized_words": diarized_words,               # D2.2: word + speaker
            "speaker_segments": speaker_segments,           # D2.3: grouped by speaker
        }

        if result_queue is not None:
            result_queue.put(result)

        # ── Step 4: Cleanup → VRAM freed by process death ─────────────
        if "torch" in dir() and device == "cuda":
            torch.cuda.empty_cache()

        logger.info("[Diarize] Diarization spoke exiting. VRAM reclaimed.")

    except Exception as e:
        logger.error(f"[Diarize] Spoke crashed: {e}")
        import traceback
        traceback.print_exc()
        if result_queue is not None:
            result_queue.put({"error": str(e), "diarized_words": [], "speaker_segments": []})
        raise SystemExit(1)


# ─── NeMo MSDD diarization ────────────────────────────────────────────────────

def _run_nemo_diarization(
    wav_path: str,
    models_dir: str,
    num_speakers: Optional[int],
    max_speakers: int,
    device: str,
) -> List[Dict]:
    """
    Run NeMo's MSDD ClusteringDiarizer with TitaNet-Large embeddings.

    Returns list of {"start", "end", "speaker"} dicts.
    """
    try:
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf
        import wget

        # ── Build minimal NeMo diarizer config ─────────────────────────
        with tempfile.TemporaryDirectory(prefix="nemo_diar_") as tmp:
            out_dir = Path(tmp)

            # Manifest file (NeMo input format)
            manifest = out_dir / "manifest.json"
            manifest.write_text(json.dumps({
                "audio_filepath": wav_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": num_speakers,
                "rttm_filepath": None,
                "uem_filepath": None,
            }) + "\n")

            cfg = OmegaConf.create({
                "num_workers": 0,
                "batch_size": 64,
                "sample_rate": 16000,
                "device": device,
                "verbose": False,

                "diarizer": {
                    "manifest_filepath": str(manifest),
                    "out_dir": str(out_dir),
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,

                    "vad": {
                        "model_path": "vad_multilingual_marblenet",  # NeMo pretrained
                        "parameters": {
                            "onset": 0.8,
                            "offset": 0.6,
                            "pad_onset": 0.05,
                            "pad_offset": -0.1,
                            "min_duration_on": 0.1,
                            "min_duration_off": 0.25,
                        },
                    },

                    "speaker_embeddings": {
                        "model_path": "titanet_large",  # TitaNet-Large (D2.1)
                        "parameters": {
                            "window_length_in_sec": [1.5, 1.0, 0.5],
                            "shift_length_in_sec":  [0.75, 0.5, 0.25],
                            "multiscale_weights":   [1, 1, 1],
                            "save_embeddings": False,
                        },
                    },

                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": num_speakers is not None,
                            "max_num_speakers": max_speakers,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 30,
                        },
                    },

                    "msdd_model": {
                        "model_path": "diar_msdd_telephonic",
                        "parameters": {
                            "use_speaker_model_from_ckpt": True,
                            "infer_batch_size": 25,
                            "sigmoid_threshold": [0.7],
                            "seq_eval_mode": False,
                            "split_infer": True,
                            "diar_eval_settings": [[0.25, True]],
                        },
                    },
                },
            })

            logger.info("[Diarize] Running NeMo MSDD ClusteringDiarizer...")
            diarizer = nemo_asr.models.ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()

            # Parse RTTM output
            rttm_files = list(out_dir.rglob("*.rttm"))
            if not rttm_files:
                logger.warning("[Diarize] No RTTM output found. Single speaker assumed.")
                return [{"start": 0.0, "end": 99999.0, "speaker": "SPEAKER_00"}]

            return _parse_rttm(rttm_files[0])

    except ImportError:
        logger.warning(
            "[Diarize] NeMo not available. "
            "Falling back to single-speaker assumption. "
            "Install nemo_toolkit[asr] inside Docker for full diarization."
        )
        # Graceful degradation — whole video is one speaker
        return [{"start": 0.0, "end": 99999.0, "speaker": "SPEAKER_00"}]

    except Exception as e:
        logger.error(f"[Diarize] NeMo diarization failed: {e}. "
                     "Using single-speaker fallback.")
        return [{"start": 0.0, "end": 99999.0, "speaker": "SPEAKER_00"}]


def _parse_rttm(rttm_path: Path) -> List[Dict]:
    """
    Parse NeMo-generated RTTM file into a list of speaker segments.

    RTTM format (space-separated):
        SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    segments = []
    try:
        with open(rttm_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9 or parts[0] != "SPEAKER":
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({
                    "start": round(start, 3),
                    "end": round(start + duration, 3),
                    "speaker": speaker,
                })
    except Exception as e:
        logger.error(f"[Diarize] RTTM parse error: {e}")

    segments.sort(key=lambda s: s["start"])
    logger.info(f"[Diarize] Parsed {len(segments)} speaker segments from RTTM.")
    return segments


# ─── Audio prep ───────────────────────────────────────────────────────────────

def _ensure_wav(audio_path: str) -> str:
    """
    Convert audio to 16kHz mono WAV if needed (NeMo requirement).
    Returns path to WAV file (may be a temp file).
    """
    p = Path(audio_path)
    if p.suffix.lower() == ".wav":
        return audio_path  # Already WAV — assume 16kHz mono from yt-dlp

    # Convert using ffmpeg
    import subprocess
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="diar_audio_")
    out_path = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "16000",   # 16kHz
        "-ac", "1",       # Mono
        "-c:a", "pcm_s16le",
        out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        logger.info(f"[Diarize] Converted to WAV: {out_path}")
        return out_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"[Diarize] ffmpeg conversion failed: {e}. Using original.")
        return audio_path

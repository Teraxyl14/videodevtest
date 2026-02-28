"""
Audio Spoke — Spoke A (Ears)
============================
Ephemeral GPU process for Automatic Speech Recognition.
Uses Parakeet TDT 0.6B v2 via NeMo toolkit.

Lifecycle: Spawn → Load Model → Transcribe → Return → Die
VRAM Usage: ~2.5GB peak
VRAM Reclamation: Guaranteed via process termination

References:
    - GPU VRAM Orchestration and IPC.txt, Section 5
    - Objective D1.1-D1.4 (Transcription)
"""

import logging
import json
from typing import Optional

logger = logging.getLogger("Spoke-ASR")


def run_asr_spoke(
    audio_path: str,
    models_dir: str,
    result_queue=None,
    device: str = "cuda",
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
):
    """
    Ephemeral ASR spoke process.

    This function runs in its own process (via multiprocessing.spawn).
    It initializes CUDA, loads the ASR model, transcribes, and exits.
    On exit, the OS kernel reclaims all VRAM.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC)
        models_dir: Directory for model weight cache
        result_queue: multiprocessing.Queue to return results
        device: "cuda" for GPU, "cpu" for CPU fallback
        model_name: HuggingFace model identifier
    """
    import os
    os.environ["TRANSFORMERS_CACHE"] = models_dir
    os.environ["HF_HOME"] = models_dir

    try:
        import torch

        # ---- 1. Initialize CUDA Context ----
        if device == "cuda" and torch.cuda.is_available():
            gpu_device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"ASR Spoke initialized on {gpu_name} ({vram_total:.1f}GB)")
        else:
            gpu_device = torch.device("cpu")
            logger.info("ASR Spoke running on CPU (fallback mode)")

        # ---- 2. Load Parakeet Model ----
        logger.info(f"Loading ASR model: {model_name}")

        try:
            # Try NeMo first (preferred for Parakeet)
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
                map_location=gpu_device
            )
            model.eval()

            # ---- 3. Transcribe ----
            logger.info(f"Transcribing: {audio_path}")

            # NeMo transcription with word-level timestamps
            transcriptions = model.transcribe(
                [audio_path],
                timestamps=True,
                batch_size=1
            )

            # Extract result
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                if hasattr(transcriptions[0], 'text'):
                    transcript_text = transcriptions[0].text
                    # Extract word timestamps if available
                    word_timestamps = []
                    if hasattr(transcriptions[0], 'timestep') and transcriptions[0].timestep:
                        for word_info in transcriptions[0].timestep.get('word', []):
                            word_timestamps.append({
                                'word': word_info.get('word', ''),
                                'start': word_info.get('start_offset', 0),
                                'end': word_info.get('end_offset', 0)
                            })
                else:
                    transcript_text = str(transcriptions[0])
                    word_timestamps = []
            else:
                transcript_text = str(transcriptions)
                word_timestamps = []

            result = {
                "text": transcript_text,
                "word_timestamps": word_timestamps,
                "model": model_name,
                "device": str(gpu_device)
            }

        except ImportError:
            # Fallback: Use transformers if NeMo not available
            logger.warning("NeMo not available. Using transformers fallback.")
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import librosa

            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(gpu_device)

            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

            with torch.no_grad():
                predicted_ids = model.generate(**inputs, return_timestamps=True)
                transcript_text = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]

            result = {
                "text": transcript_text,
                "word_timestamps": [],
                "model": model_name,
                "device": str(gpu_device)
            }

        logger.info(f"Transcription complete: {len(transcript_text)} chars")

        # ---- 4. Return result via Queue ----
        if result_queue is not None:
            result_queue.put(result)

        # ---- 5. Cleanup (process death handles the rest) ----
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        logger.info("ASR Spoke exiting. VRAM will be reclaimed by OS.")

    except Exception as e:
        logger.error(f"ASR Spoke crashed: {e}")
        import traceback
        traceback.print_exc()
        if result_queue is not None:
            result_queue.put({"error": str(e)})
        raise SystemExit(1)

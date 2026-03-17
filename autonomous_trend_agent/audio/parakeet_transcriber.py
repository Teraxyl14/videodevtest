"""
Parakeet TDT Transcriber - V2
=============================
Ultra-fast transcription using NVIDIA's Parakeet TDT 0.6B/1.1B model.

Performance:
- 3300x Real-Time Factor (RTF) vs ~100x for Whisper
- Native word-level timestamps (no forced alignment)
- 0.6GB VRAM for 0.6B model

Features:
- Token-and-Duration Transducer architecture
- Native timestamp generation
- Speaker diarization via NeMo TitaNet
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json

import torch
import autonomous_trend_agent.core.nemo_compat  # Apply NeMo/Datasets patches

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """A single word with timing."""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass 
class Segment:
    """A segment of transcribed speech."""
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: List[Segment]
    language: str = "en"
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence}
                        for w in seg.words
                    ]
                }
                for seg in self.segments
            ]
        }
    
    def get_words(self) -> List[Dict]:
        """Get flat list of all words with timing."""
        words = []
        for seg in self.segments:
            for word in seg.words:
                words.append({
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "speaker": seg.speaker
                })
        return words


class ParakeetTranscriber:
    """
    Ultra-fast transcription using NVIDIA Parakeet TDT.
    
    Parakeet TDT uses Token-and-Duration Transducer architecture
    which outputs both tokens and their durations in a single pass,
    eliminating the need for CTC alignment post-processing.
    
    Usage:
        transcriber = ParakeetTranscriber()
        result = transcriber.transcribe("audio.wav")
        words = result.get_words()  # Word-level timestamps
    """
    
    # Available model sizes
    MODELS = {
        "0.6b": "nvidia/parakeet-tdt-0.6b-v3",    # Fastest, ~0.6GB VRAM
        "1.1b": "nvidia/parakeet-tdt-1.1b",    # More accurate, ~1.2GB VRAM
    }
    
    def __init__(
        self,
        model_size: str = "1.1b",
        device: str = "cuda",
        enable_diarization: bool = False
    ):
        """
        Initialize Parakeet transcriber.
        
        Args:
            model_size: "0.6b" (faster) or "1.1b" (more accurate)
            device: "cuda" or "cpu"
            enable_diarization: Enable speaker diarization via TitaNet
        """
        self.model_size = model_size
        self.model_size = model_size
        # Transformers pipeline expects integer device ID for GPU (0), not "cuda" string
        self.device = 0 if torch.cuda.is_available() and device == "cuda" else "cpu"
        # If device was passed as "cpu" explicitly
        if device == "cpu":
            self.device = -1 # -1 for CPU in pipeline
            
        self.enable_diarization = enable_diarization
        
        self._model = None
        self._diarizer = None
        self._loaded = False
        
        if model_size not in self.MODELS:
            raise ValueError(f"Unknown model size: {model_size}. Available: {list(self.MODELS.keys())}")
        
        self.model_name = self.MODELS[model_size]
    
    def _ensure_loaded(self):
        """Lazy load model on first use. Falls back to Whisper if NeMo unavailable."""
        if self._loaded:
            return
        
        # Try NeMo Parakeet first
        try:

            # --- BEGIN MONKEY-PATCH for Datasets >= 3.0 (NeMo Fix) ---
            import sys
            import types
            try:
                import datasets.distributed
            except ImportError:
                dummy_distributed = types.ModuleType("datasets.distributed")
                def split_dataset_by_node(dataset, rank, world_size):
                    return dataset 
                dummy_distributed.split_dataset_by_node = split_dataset_by_node
                sys.modules["datasets.distributed"] = dummy_distributed
            # --- END MONKEY-PATCH ---

            # Monkey-patch HfFolder for NeMo compatibility (huggingface_hub > 0.20 break)
            import huggingface_hub
            if not hasattr(huggingface_hub, "HfFolder"):
                from huggingface_hub import HfApi
                class MockHfFolder:
                    @staticmethod
                    def get_token(): return HfApi().token
                    @staticmethod
                    def save_token(token): pass
                huggingface_hub.HfFolder = MockHfFolder
            
            if not hasattr(huggingface_hub, "ModelFilter"):
                class MockModelFilter:
                    def __init__(self, **kwargs): pass
                huggingface_hub.ModelFilter = MockModelFilter
            
            # --- PATCH: Fix hf_hub_download 'use_auth_token' error in NeMo ---
            # NeMo passes 'use_auth_token' which was removed in huggingface_hub v0.20+
            original_hf_hub_download = huggingface_hub.hf_hub_download
            
            def patched_hf_hub_download(*args, **kwargs):
                if 'use_auth_token' in kwargs:
                    # Rename to 'token' or remove if None
                    token = kwargs.pop('use_auth_token')
                    if token and 'token' not in kwargs:
                        kwargs['token'] = token
                return original_hf_hub_download(*args, **kwargs)
            
            huggingface_hub.hf_hub_download = patched_hf_hub_download
            # ------------------------------------------------------------------

            import nemo.collections.asr as nemo_asr
            
            # --- BEGIN REPRODUCIBLE MONKEY-PATCH FOR NEMO TIMESTAMPS=TRUE BUG ---
            # This fixes "NameError: name 'copy' is not defined" in RNNT decoding
            try:
                import copy
                import sys
                # Force import to ensure it's in sys.modules
                import nemo.collections.asr.parts.submodules.rnnt_decoding as rnnt_decoding
                # Inject copy into the module global namespace
                setattr(rnnt_decoding, 'copy', copy)
                if 'nemo.collections.asr.parts.submodules.rnnt_decoding' in sys.modules:
                    setattr(sys.modules['nemo.collections.asr.parts.submodules.rnnt_decoding'], 'copy', copy)
                logger.info("Successfully patched NeMo rnnt_decoding with missing 'copy' module")
            except Exception as e:
                logger.debug(f"NeMo optional patch skipped or failed: {e}")
            # --- END MONKEY-PATCH ---
            
            logger.info(f"Loading Parakeet TDT model: {self.model_name}")
            self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Apply memory optimizations for long audio (Deep Research finding)
            try:
                # 1. Local relative attention caps quadratic memory scaling
                self._model.change_attention_model(
                    self._model.cfg.encoder.get('self_attention_model', 'rel_pos_local_attn'),
                    update_config=True
                )
                
                # If that failed and we need strict rel_pos_local_attn:
                if not hasattr(self._model.encoder, 'attention_type') or self._model.encoder.attention_type != 'rel_pos_local_attn':
                    self._model.change_attention_model('rel_pos_local_attn', update_config=True)

                # 2. Convolutional sub-sampling chunking to prevent initial layer OOM
                if hasattr(self._model, 'change_subsampling_conv_chunking_factor'):
                    self._model.change_subsampling_conv_chunking_factor(1)
                logger.info("Applied TDT memory optimizations (local attention + chunking)")
            except Exception as e:
                logger.warning(f"Could not apply memory optimizations: {e}")
                
            logger.info(f"Parakeet TDT loaded on {self.device}")
            self._backend = "nemo"
            if self.enable_diarization:
                self._load_diarizer()
            self._loaded = True
            return
        except (ImportError, ModuleNotFoundError) as e:
            nemo_asr = None
            logger.warning(f"NeMo import failed: {e}, falling back to Whisper...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"NeMo unavailable ({e}), falling back to Whisper...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Fallback: Whisper via transformers (Optimized Zero-Copy Load)
        try:
            from transformers import pipeline as hf_pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
            import autonomous_trend_agent.core.nemo_compat
            import os
            import logging
            from pathlib import Path
            
            # REPORT FINDING: INT8/Int4 is unstable on Blackwell (sm_120). 
            # Force Float16 for 100% stability.
            whisper_model_id = "distil-whisper/distil-large-v3"
            logger.info(f"Loading Whisper fallback: {whisper_model_id} (Float16 Zero-Copy)")
            
            # Use AutoModel to control loading path and prevent CPU RAM explosion
            # low_cpu_mem_usage=True enables Meta Device init (sharded loading)
            logger.debug(f"Loading Whisper model on device={self.device} (Zero-Copy)")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="flash_attention_2",  # CRITICAL: For 16GB VRAM
                # If device is 0, use "cuda:0", else "cpu"
                device_map="cuda:0" if self.device == 0 else "cpu"
            )

            logger.debug("Loading Whisper processor")
            processor = AutoProcessor.from_pretrained(whisper_model_id)

            logger.debug("Initializing HF pipeline with pre-loaded model")
            self._whisper_pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=15,
                batch_size=1,  # Down from 4 to prevent CPU RAM OOM on large files
                torch_dtype=torch.float16,
                device="cuda:0" if self.device == 0 else -1
            )
            
            logger.info(f"Whisper loaded on GPU (Float16 Zero-Copy)")
            self._backend = "whisper"
            self._loaded = True
        except Exception as e:
             raise ImportError(f"Both NeMo and Whisper failed: {e}")

    def transcribe(self, audio_path: str, return_timestamps: bool = True) -> TranscriptionResult:
        """Transcribe audio/video file with automatic audio extraction."""
        self._ensure_loaded()
        
        input_path = Path(audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
            
        # Check if we need to extract audio (if input is video)
        # We'll use a temp file for safety and robustness
        temp_audio = None
        target_path = input_path
        
        import mimetypes
        mime_type, _ = mimetypes.guess_type(input_path)
        is_video = mime_type and mime_type.startswith('video')
        
        # Explicitly extract audio if it's a video file or unknown
        if is_video or input_path.suffix.lower() in ['.mp4', '.mov', '.mkv', '.avi']:
             import tempfile
             import subprocess
             
             temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
             os.close(temp_fd)
             temp_audio = Path(temp_audio_path)
             
             temp_audio = Path(temp_audio_path)
             
             logger.info(f"Extracting audio from video: {input_path} -> {temp_audio}")
             
             # Extract 16kHz mono WAV for optimal compatibility
             cmd = [
                 'ffmpeg', '-y',
                 '-i', str(input_path),
                 '-vn',
                 '-acodec', 'pcm_s16le',
                 '-ar', '16000',
                 '-ac', '1',
                 str(temp_audio)
             ]
             try:
                 subprocess.run(cmd, check=True, capture_output=True)
                 logger.debug("FFmpeg audio extraction complete")
                 target_path = temp_audio
             except subprocess.CalledProcessError as e:
                 logger.error(f"FFmpeg extraction failed: {e.stderr.decode()}")
                 if temp_audio.exists():
                     temp_audio.unlink()
                 raise RuntimeError(f"Failed to extract audio from video: {e}")

        try:
            logger.info(f"Transcribing: {target_path.name} (backend: {self._backend})")
            if self._backend == "whisper":
                res = self._transcribe_whisper(str(target_path), return_timestamps)
            else:
                res = self._transcribe_nemo(str(target_path), return_timestamps)
            
            # If no duration, try getting it from original file
            if res.duration == 0.0:
                 res.duration = self._get_audio_duration(str(input_path))
                 
            return res
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return TranscriptionResult(text="", segments=[], duration=0.0)
            
        finally:
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()

    def _transcribe_nemo(self, audio_path: str, return_timestamps: bool = True) -> TranscriptionResult:
        """
        Transcribe using NeMo Parakeet TDT model, with safe VRAM chunking.
        
        ARCHITECTURE NOTE: 
        The raw NeMo Parakeet model memory-scales quadratically with audio length.
        To maintain the strict 16GB VRAM constraint on the RTX 5080, this method 
        splits the source audio into 60-second chunks using FFmpeg, processes 
        them sequentially through the model, and mathematically stitches the 
        timestamps back together to form one cohesive, absolute-time timeline.
        """
        import tempfile
        import subprocess
        from pathlib import Path
        import gc

        try:
            duration = self._get_audio_duration(audio_path)
            # 60s chunks for absolute stability and lower memory pressure
            chunk_length_s = 60.0  
            num_chunks = int(duration // chunk_length_s) + 1 if duration > 0 else 1
            
            all_words = []
            full_text = []

            with tempfile.TemporaryDirectory() as temp_dir:
                for i in range(num_chunks):
                    start_time = i * chunk_length_s
                    if duration > 0 and start_time >= duration:
                        break
                        
                    chunk_duration = min(chunk_length_s, duration - start_time) if duration > 0 else chunk_length_s
                    if chunk_duration <= 0:
                        break

                    chunk_path = str(Path(temp_dir) / f"chunk_{i}.wav")
                    logger.info(f"Processing chunk {i+1}/{num_chunks} ({start_time:.1f}s to {start_time+chunk_duration:.1f}s)...")
                    
                    cmd = [
                        'ffmpeg', '-y', '-i', audio_path, 
                        '-ss', str(start_time), '-t', str(chunk_duration),
                        '-c', 'copy', chunk_path
                    ]
                    subprocess.run(cmd, capture_output=True, check=True)
                    
                    with torch.no_grad():
                        # Configure stable legacy fallback
                        from omegaconf import open_dict
                        decoding_cfg = self._model.cfg.decoding
                        with open_dict(decoding_cfg):
                            decoding_cfg.preserve_alignments = True
                            # DISABLE built-in compute_timestamps to avoid ValueError mismatch crash
                            decoding_cfg.compute_timestamps = False
                            decoding_cfg.word_seperator = " "
                        
                        self._model.change_decoding_strategy(decoding_cfg)
                        hypotheses = self._model.transcribe([chunk_path], return_hypotheses=True)
                        
                        if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
                            hypotheses = hypotheses[0]
                            
                        hypothesis = hypotheses[0]
                        chunk_text = hypothesis.text if hasattr(hypothesis, 'text') else ""
                        parsed_word_timestamps = []
                        
                        # TDT Algorithm typically utilizes an 8x internal downsampling window constraint.
                        # We must calculate this exact time_stride ratio to accurately cross-multiply 
                        # the raw token emissions back into absolute seconds relative to the chunk.
                        try:
                            window_stride = self._model.cfg.preprocessor.window_stride
                            time_stride = 8 * window_stride # TDT has 8x downsampling
                        except:
                            time_stride = 0.08
                        
                        # Manually parse timestep even if compute_timestamps is False
                        # NeMo 1.23 throws a ValueError if compute_timestamps=True due to a strict array 
                        # length validation failing on edge cases. Turning it off disables the crash, 
                        # but the raw token timestep data is still emitted in the object. We parse it here.
                        if hasattr(hypothesis, 'timestep') and hypothesis.timestep:
                            # TDT often provides 'word' timestamps directly in timestep
                            if 'word' in hypothesis.timestep:
                                raw_words = hypothesis.timestep['word']
                                for stamp in raw_words:
                                    word_text = stamp.get('word', '')
                                    # Use offsets if present, else use start/end directly
                                    start_off = stamp.get('start_offset', stamp.get('start', 0))
                                    end_off = stamp.get('end_offset', stamp.get('end', 0))
                                    
                                    # Convert to absolute video time
                                    s_time = start_off * time_stride + start_time
                                    e_time = end_off * time_stride + start_time
                                    
                                    parsed_word_timestamps.append(Word(
                                        text=word_text,
                                        start=round(float(s_time), 3),
                                        end=round(float(e_time), 3),
                                        confidence=1.0
                                    ))
                            elif 'char' in hypothesis.timestep:
                                # Fallback to char-level if word is missing
                                logger.debug(f"Chunk {i}: Word timesteps missing, falling back to char-level.")
                                pass # We'll just aggregate text later
                        
                        # If no timestamps were extracted but we have text, create a simple uniform distribution
                        if not parsed_word_timestamps and chunk_text:
                            logger.warning(f"Chunk {i}: No timestamps found in timestep. Distributing words uniformly.")
                            words = chunk_text.split()
                            if words:
                                word_dur = chunk_duration / len(words)
                                for idx, w in enumerate(words):
                                    parsed_word_timestamps.append(Word(
                                        text=w,
                                        start=round(start_time + idx * word_dur, 3),
                                        end=round(start_time + (idx + 1) * word_dur, 3),
                                        confidence=0.5
                                    ))

                        all_words.extend(parsed_word_timestamps)
                        full_text.append(chunk_text)
                        
                        # Aggressive VRAM cleanup
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Create final result
            combined_text = " ".join(full_text)
            segments = self._create_segments(combined_text, all_words)
            
            if self.enable_diarization:
                segments = self._add_speaker_labels(audio_path, segments)
                
            return TranscriptionResult(
                text=combined_text,
                segments=segments,
                duration=duration
            )

        except Exception as e:
            logger.error(f"NeMo transcription failed: {e}", exc_info=True)
            raise

        except Exception as e:
            logger.error(f"NeMo transcription failed: {e}")
            raise

    def _transcribe_whisper(self, audio_path: str, return_timestamps: bool = True) -> TranscriptionResult:
        """Transcribe using Whisper via transformers pipeline."""
        try:
            logger.debug("Starting Whisper pipeline execution")
            result = self._whisper_pipe(
                audio_path,
                return_timestamps="word" if return_timestamps else None,
                generate_kwargs={"language": "en", "task": "transcribe"},
            )
            logger.debug("Whisper pipeline execution finished")
            
            text = result.get("text", "").strip()
            words = []
            
            chunks = result.get("chunks", [])
            if return_timestamps and chunks:
                for chunk in chunks:
                    ts = chunk.get("timestamp", (0.0, 0.0))
                    if ts and len(ts) == 2 and ts[0] is not None:
                        # Handle case where end timestamp is None (open-ended)
                        start_t = float(ts[0])
                        end_t = float(ts[1]) if ts[1] is not None else start_t + 0.1
                        
                        word = Word(
                            text=chunk.get("text", "").strip(),
                            start=start_t,
                            end=end_t,
                            confidence=1.0
                        )
                        if word.text:
                            words.append(word)
            
            # Create segments
            segments = self._create_segments(text, words)
            duration = self._get_audio_duration(audio_path)
            
            return TranscriptionResult(
                text=text,
                segments=segments,
                language="en",
                duration=duration
            )
        except Exception as e:
            logger.error(f"Whisper processing failed: {e}")
            raise

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds. Robust to format."""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            if info.duration > 0:
                return info.duration
        except:
             pass
             
        # Fallback to FFmpeg probe
        try:
            import subprocess
            cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception:
            return 0.0
    
    def _create_segments(
        self,
        text: str,
        words: List[Word],
        max_words_per_segment: int = 10
    ) -> List[Segment]:
        """Group words into segments."""
        if not words:
            # Fallback: create single segment without word timing
            return [Segment(text=text, start=0.0, end=0.0, words=[])]
        
        segments = []
        current_words = []
        
        for word in words:
            current_words.append(word)
            
            # Split on punctuation or max words
            if (len(current_words) >= max_words_per_segment or 
                word.text.endswith(('.', '!', '?', ','))):
                
                seg_text = " ".join(w.text for w in current_words)
                segment = Segment(
                    text=seg_text,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    words=current_words.copy()
                )
                segments.append(segment)
                current_words = []
        
        # Don't forget remaining words
        if current_words:
            seg_text = " ".join(w.text for w in current_words)
            segment = Segment(
                text=seg_text,
                start=current_words[0].start,
                end=current_words[-1].end,
                words=current_words
            )
            segments.append(segment)
        
        return segments
    
    def _add_speaker_labels(
        self,
        audio_path: str,
        segments: List[Segment]
    ) -> List[Segment]:
        """Add speaker labels to segments using TitaNet."""
        if not self._diarizer:
            return segments
        
        try:
            # Run diarization
            diarization = self._diarizer.diarize([audio_path])
            
            # Map segments to speakers based on timestamp overlap
            for segment in segments:
                mid_time = (segment.start + segment.end) / 2
                
                for speaker_segment in diarization:
                    if speaker_segment['start'] <= mid_time <= speaker_segment['end']:
                        segment.speaker = speaker_segment['speaker']
                        break
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
        
        return segments
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except:
            return 0.0
    
    def unload(self):
        """Unload model to free VRAM."""
        if self._model:
            del self._model
            self._model = None
        if self._diarizer:
            del self._diarizer
            self._diarizer = None
        
        if hasattr(self, '_whisper_pipe') and self._whisper_pipe:
            del self._whisper_pipe
            self._whisper_pipe = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._loaded = False
        logger.info("Transcriber unloaded")

    def save_transcript(self, result: TranscriptionResult, output_path: str):
        """Save transcription to JSON file."""
        # Use module-level function
        save_transcript(result, output_path)


def save_transcript(result: TranscriptionResult, output_path: str):
    """Save transcription to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Transcript saved to: {output_path}")


def extract_audio_from_video(video_path: str, output_path: str) -> str:
    """Extract audio from video for transcription."""
    import subprocess
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"Audio extracted to: {output_path}")
    return output_path


# Convenience function for direct use
def transcribe_video(video_path: str, save_path: Optional[str] = None) -> TranscriptionResult:
    """
    Transcribe audio from a video file.
    
    Args:
        video_path: Path to video file
        save_path: Optional path to save JSON transcript
    
    Returns:
        TranscriptionResult with word-level timestamps
    """
    import tempfile
    
    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    
    try:
        extract_audio_from_video(video_path, audio_path)
        
        # Transcribe
        transcriber = ParakeetTranscriber()
        result = transcriber.transcribe(audio_path)
        
        # Save if requested
        if save_path:
            save_transcript(result, save_path)
        
        return result
        
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

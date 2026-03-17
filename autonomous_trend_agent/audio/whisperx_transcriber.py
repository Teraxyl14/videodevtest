import os
import gc
import torch
import whisperx

def execute_audio_alignment_spoke(audio_filepath: str, hf_auth_token: str) -> dict:
    """
    Ephemeral Spoke execution for WhisperX. Sequentially loads the ASR,
    Forced Alignment, and Diarization models, clearing VRAM between stages.
    """
    execution_device = "cuda" if torch.cuda.is_available() else "cpu"
    # Leverage the RTX 5080's Tensor Cores with half-precision floating point
    tensor_precision = "float16" if execution_device == "cuda" else "float32"

    # Stage 1: Initial Transcription via Distil-Whisper Large-v3 
    # Generates accurate text but imprecise, utterance-level timestamps
    asr_pipeline = whisperx.load_model(
        "distil-large-v3", 
        execution_device, 
        compute_type=tensor_precision
    )
    raw_audio_waveform = whisperx.load_audio(audio_filepath)
    transcription_result = asr_pipeline.transcribe(raw_audio_waveform, batch_size=16)

    # Aggressively reclaim VRAM prior to loading the phoneme-based model
    del asr_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2: Forced Alignment via wav2vec2 
    # Maps orthographic transcripts to the acoustic signal for millisecond precision
    try:
        alignment_model, alignment_metadata = whisperx.load_align_model(
            language_code=transcription_result.get("language", "en"), 
            device=execution_device
        )

        aligned_segments = whisperx.align(
            transcription_result["segments"], 
            alignment_model, 
            alignment_metadata, 
            raw_audio_waveform, 
            execution_device, 
            return_char_alignments=False
        )
        
        # Reclaim VRAM prior to Speaker Diarization
        del alignment_model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WhisperX] Alignment fallback: {e}")
        aligned_segments = transcription_result
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 3: Speaker Diarization via Pyannote 4.0 
    # Partitions the audio stream into homogeneous speaker segments
    if not hf_auth_token:
        print("[WhisperX] HF_TOKEN not provided! Skipping Pyannote Diarization.")
        fully_attributed_result = aligned_segments
    else:
        try:
            diarization_pipeline = whisperx.DiarizationPipeline(
                use_auth_token=hf_auth_token, 
                device=execution_device
            )
            speaker_segments = diarization_pipeline(audio_filepath, min_speakers=1, max_speakers=5)

            # Project the speaker IDs onto the millisecond-aligned words
            fully_attributed_result = whisperx.assign_word_speakers(speaker_segments, aligned_segments)

            # Final cleanup before the Spoke process terminates
            del diarization_pipeline
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WhisperX] Diarization failure: {e}")
            fully_attributed_result = aligned_segments

    return fully_attributed_result

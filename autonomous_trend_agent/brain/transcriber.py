"""
GPU-Accelerated Speech Transcription using Whisper
Provides word-level timestamps for animated captions
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Word:
    """Transcribed word with timing"""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0


class DeepTranscriber:
    """
    GPU-accelerated speech-to-text transcription using Whisper.
    Optimized for the Golden Stack pipeline.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_size = model_size
        self._model = None
        
        print(f"[Transcriber] Initializing Whisper ({model_size}, {device})")
    
    def _ensure_loaded(self):
        """Lazy load Whisper model"""
        if self._model is not None:
            return
        
        try:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            print(f"[Transcriber] Whisper {self.model_size} loaded successfully")
        except ImportError:
            print("[Transcriber] WARNING: openai-whisper not installed, using transformers pipeline")
            from transformers import pipeline
            self._model = pipeline(
                "automatic-speech-recognition",
                model=f"openai/whisper-{self.model_size}",
                device=0 if self.device == "cuda" else -1,
                return_timestamps="word"
            )
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio/video file
            language: Optional language code (e.g., 'en')
            
        Returns:
            Dict with 'text' and 'words' (list of Word objects with timestamps)
        """
        self._ensure_loaded()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"[Transcriber] Transcribing: {audio_path.name}")
        
        # Check if using native whisper or transformers
        if hasattr(self._model, 'transcribe'):
            # Native openai-whisper
            result = self._model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True,
                fp16=(self.device == "cuda")
            )
            
            return self._parse_whisper_result(result)
        else:
            # Transformers pipeline
            result = self._model(str(audio_path), return_timestamps="word")
            return self._parse_transformers_result(result)
    
    def _parse_whisper_result(self, result: Dict) -> Dict:
        """Parse native Whisper result"""
        words = []
        
        for segment in result.get("segments", []):
            for word_data in segment.get("words", []):
                words.append({
                    "word": word_data.get("word", "").strip(),
                    "start_time": word_data.get("start", 0.0),
                    "end_time": word_data.get("end", 0.0),
                    "confidence": word_data.get("probability", 1.0)
                })
        
        return {
            "text": result.get("text", ""),
            "words": words,
            "language": result.get("language", "unknown")
        }
    
    def _parse_transformers_result(self, result: Dict) -> Dict:
        """Parse transformers pipeline result"""
        words = []
        
        # Transformers format: {'text': str, 'chunks': [{'text': str, 'timestamp': (start, end)}]}
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp", (0.0, 0.0))
            words.append({
                "word": chunk.get("text", "").strip(),
                "start_time": timestamp[0] if timestamp[0] is not None else 0.0,
                "end_time": timestamp[1] if timestamp[1] is not None else 0.0,
                "confidence": 1.0
            })
        
        return {
            "text": result.get("text", ""),
            "words": words,
            "language": "unknown"
        }
    
    def save_transcript(self, transcript: Dict, output_path: str):
        """Save transcript to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"[Transcriber] Saved transcript to: {output_path}")
    
    def unload(self):
        """Free GPU memory"""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            print("[Transcriber] Model unloaded, VRAM freed")

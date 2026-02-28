"""
Unified Pipeline Orchestrator
Single entry point for the complete short-form content generation pipeline

Flow:
1. Trend Discovery → Identify viral topics
2. Video Analysis → Find best segments using Qwen3-VL
3. Tracking → Detect subjects with YOLOv11n
4. Editing → Reframe with GPU pipeline + effects
5. Export → Generate shorts with metadata
"""

import os
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict, field
import multiprocessing as mp

# Core Engine Imports
from autonomous_trend_agent.core.vram_locker import start_locker
from autonomous_trend_agent.core.viral_engine import ViralEngine
from dataclasses import asdict


# Pipeline Module Imports
try:
    from autonomous_trend_agent.pipeline.thumbnail_extractor import ThumbnailExtractor
    from autonomous_trend_agent.pipeline.quality_assurance import QualityAssurance
except ImportError:
    ThumbnailExtractor = None
    QualityAssurance = None

try:
    from autonomous_trend_agent.core.blackwell_hub import BlackwellHub
except ImportError:
    BlackwellHub = None


# Import Context Managers and Server
# Import Context Managers
try:
    from autonomous_trend_agent.core.gpu_pipeline import TransitionContextManager
except ImportError:
    # Fallback/Mock for dev without full env
    class TransitionContextManager:
        def __init__(self, name): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    # Input
    video_path: str = ""
    transcript_path: Optional[str] = None
    trend_topic: Optional[str] = None
    
    # Output
    output_dir: str = "./output"
    
    # Processing options
    num_shorts: int = 4
    short_duration_range: Tuple[int, int] = (55, 90)  # seconds
    target_resolution: tuple = (1080, 1920)  # 9:16 vertical
    
    # Module toggles
    use_gpu_reframing: bool = True
    use_yolo_tracking: bool = True
    use_qwen_analysis: bool = True
    use_gemini_analysis: bool = True  # Re-enabled: Gemini Integration
    use_effects: bool = True
    use_captions: bool = True
    use_active_speaker: bool = True  # Follow the speaking person
    use_llm_director: bool = True    # Re-enabled: LLM Director (Gemini)
    use_boredom_detector: bool = True # Detect low-info segments and auto-zoom
    caption_style: str = "tiktok"  # tiktok, hormozi, mrbeast, minimal, neon
    
    # Performance
    batch_size: int = 50
    device: str = "cuda"


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    success: bool
    video_path: str
    output_dir: str
    shorts_created: List[Dict]
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ProgressCallback:
    """Base class for progress callbacks"""
    
    def on_stage_start(self, stage: str, total_steps: int = 0):
        print(f"\n{'='*60}")
        print(f"[STAGE] {stage}")
        print(f"{'='*60}")
    
    def on_step(self, step: str, current: int = 0, total: int = 0):
        if total > 0:
            pct = current / total * 100
            print(f"  [{current}/{total}] ({pct:.0f}%) {step}")
        else:
            print(f"  → {step}")
    
    def on_stage_complete(self, stage: str, result: Any = None):
        print(f"  ✅ {stage} complete")
    
    def on_error(self, stage: str, error: str):
        print(f"  ❌ Error in {stage}: {error}")


class PipelineOrchestrator:
    """
    Orchestrates the complete short-form content generation pipeline.
    
    Connects all Golden Stack modules:
    - TrendAggregator (discovery)
    - Qwen3VideoAnalyzer (intelligence)  
    - YOLOv11Tracker (tracking)
    - HardwareReframer (editing)
    - ContextAwareEffectsEngine (effects)
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        callback: Optional[ProgressCallback] = None
    ):
        self.config = config or PipelineConfig()
        self.callback = callback or ProgressCallback()
        self.device = self.config.device
        
        # Lazy-loaded modules
        self._analyzer = None
        self._tracker = None
        self._reframer = None
        self._effects = None
        self._trend_discovery = None
        self._caption_engine = None
        self._active_speaker = None
        self._director = None
        self._transcriber = None
        self._gemini_services = None  # For AI enhancements
        self._boredom_detector = None

        # API Server
        # API Server (Removed)
        self.server_manager = None

        if os.getenv("HF_TOKEN"):
            from huggingface_hub import login
            try:
                login(token=os.getenv("HF_TOKEN"))
                print("Hugging Face logged in successfully")
            except Exception as e:
                print(f"Hugging Face login failed: {e}")

        # Initialize VRAM Locker (Persistent Process) — only if CUDA available
        self.locker_process = None
        self.ipc_handle = None
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                print("[Orchestrator] Starting VRAM Locker...")
                self.locker_process, self.ipc_handle, _ = start_locker()
                print(f"[Orchestrator] VRAM Locker Active. Handle: {self.ipc_handle}")
            except Exception as e:
                print(f"[Orchestrator] Failed to start VRAM Locker: {e}")

        # Initialize Viral Engine (Logic Layer)
        try:
            self.viral_engine = ViralEngine()
        except Exception as e:
            print(f"[Orchestrator] ViralEngine init failed (filterpy missing?): {e}")
            self.viral_engine = None


    def shutdown(self):
        """Safe shutdown of all components"""
        # Server manager removed
        pass
            
        if hasattr(self, 'locker_process') and self.locker_process:
            print("[Orchestrator] Shutting down VRAM Locker...")
            try:
                if self.locker_process.is_alive():
                    self.locker_process.terminate()
                    self.locker_process.join(timeout=2)
            except (AttributeError, TypeError, ImportError):
                # Interpreter shutdown chaos
                pass
            self.locker_process = None

    def __del__(self):
        self.shutdown()

    
    @property
    def boredom_detector(self):
        """Lazy load Boredom Detector"""
        if self._boredom_detector is None and self.config.use_boredom_detector:
            from autonomous_trend_agent.editor.retention.boredom_detector import BoredomDetector
            self._boredom_detector = BoredomDetector()
        return self._boredom_detector
    
    @property
    def analyzer(self):
        """Lazy load Qwen3 analyzer"""
        if self._analyzer is None and self.config.use_qwen_analysis:
            with TransitionContextManager("Init Qwen3 Analyzer"):
                from autonomous_trend_agent.brain.qwen3_video_analyzer import Qwen3VideoAnalyzer
                self._analyzer = Qwen3VideoAnalyzer(device=self.device)
        return self._analyzer
    
    @property
    def tracker(self):
        """Lazy load YOLO tracker"""
        if self._tracker is None and self.config.use_yolo_tracking:
            with TransitionContextManager("Init YOLO Tracker"):
                from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker
                self._tracker = YOLOv11Tracker(device=self.device, enable_pose=True)
        return self._tracker
    
    @property
    def reframer(self):
        """Lazy load reframer - STRICTLY ZeroCopyPipeline (GPU)"""
        if self._reframer is None:
            # Production-quality with per-frame Kalman tracking
            from autonomous_trend_agent.editor.zero_copy_pipeline import ZeroCopyPipeline
            self._reframer = ZeroCopyPipeline(
                target_width=self.config.target_resolution[0],
                target_height=self.config.target_resolution[1],
                batch_size=self.config.batch_size,
                device=self.device
            )
            print("[Orchestrator] Using ZeroCopyPipeline (Reference Implementation)")
        return self._reframer
    
    @property
    def effects(self):
        """Lazy load effects engine"""
        if self._effects is None and self.config.use_effects:
            from autonomous_trend_agent.effects.effects_engine import ContextAwareEffectsEngine
            self._effects = ContextAwareEffectsEngine(device=self.device)
        return self._effects
    
    @property
    def caption_engine(self):
        """Lazy load animated caption engine"""
        if self._caption_engine is None and self.config.use_captions:
            from autonomous_trend_agent.captions.animated_captions import AnimatedCaptionEngine
            self._caption_engine = AnimatedCaptionEngine(
                style=self.config.caption_style,
                device=self.device
            )
        return self._caption_engine
    
    @property
    def active_speaker(self):
        """Lazy load active speaker detector"""
        if self._active_speaker is None and self.config.use_active_speaker:
            from autonomous_trend_agent.sensors.active_speaker import ActiveSpeakerDetector
            self._active_speaker = ActiveSpeakerDetector(device=self.device)
        return self._active_speaker
    
    @property
    def director(self):
        """Lazy load LLM Director"""
        if self._director is None and self.config.use_llm_director:
            from autonomous_trend_agent.director.editor_llm import LLMDirector
            self._director = LLMDirector()
        return self._director
    
    @property
    def transcriber(self):
        """Lazy load audio transcriber — uses Parakeet TDT per spec"""
        if self._transcriber is None:
            try:
                from autonomous_trend_agent.audio.parakeet_transcriber import ParakeetTranscriber
                self._transcriber = ParakeetTranscriber(
                    model_size="1.1b",
                    device=self.device,
                    enable_diarization=False
                )
                print("[Orchestrator] Using Parakeet TDT transcriber")
            except ImportError:
                # Fallback to Whisper-based transcriber
                print("[Orchestrator] Parakeet unavailable, falling back to Whisper")
                from autonomous_trend_agent.brain.transcriber import DeepTranscriber
                self._transcriber = DeepTranscriber(model_size="large-v3", device=self.device)
        return self._transcriber
    
    @property
    def gemini_services(self):
        """Lazy load Gemini AI enhancement services"""
        if self._gemini_services is None and self.config.use_gemini_analysis:
            try:
                from autonomous_trend_agent.brain.gemini_services import GeminiServices
                self._gemini_services = GeminiServices()
            except Exception as e:
                print(f"[Orchestrator] Gemini services unavailable: {e}")
        return self._gemini_services
    
    def run(self, video_path: str, transcript_path: Optional[str] = None) -> PipelineResult:
        """
        Run the complete pipeline on a video.
        
        Args:
            video_path: Path to source video
            transcript_path: Optional path to transcript JSON
            
        Returns:
            PipelineResult with generated shorts
        """
        start_time = time.time()
        errors = []
        shorts_created = []
        analysis = None
        
        self.config.video_path = video_path
        self.config.transcript_path = transcript_path
        
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            return PipelineResult(
                success=False,
                video_path=str(video_path),
                output_dir=self.config.output_dir,
                shorts_created=[],
                duration_seconds=0,
                errors=[f"Video not found: {video_path}"]
            )
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript = None
        analysis_result = None
        
        try:
            # 1.1 Transcript
            with TransitionContextManager("Transcription Phase"):
                if self.config.transcript_path and os.path.exists(self.config.transcript_path):
                    self.callback.on_step("Loading existing transcript")
                    with open(self.config.transcript_path, 'r', encoding='utf-8') as f:
                        transcript = f.read() # Simplified, usually JSON
                elif self.config.use_captions: # Only transcribe if captions are enabled
                    self.callback.on_stage_start("Audio Transcription", 1)
                    self.callback.on_step("Transcribing video (Parakeet TDT)")
                    try:
                        transcript_path = str(output_dir / "transcript.json")
                        transcript_data = self.transcriber.transcribe(str(video_path))
                        self.transcriber.save_transcript(transcript_data, transcript_path)
                        self.config.transcript_path = transcript_path
                        transcript = json.dumps(asdict(transcript_data)) # Store as string for consistency
                        self.callback.on_stage_complete("Audio Transcription")
                    except Exception as e:
                        self.callback.on_error("Transcription", str(e))
                        transcript_path = None
                        transcript = None
                else:
                    self.callback.on_step("Transcription skipped (not enabled or no path provided)")
            
            # UNLOAD TRANSCRIBER to free VRAM for Qwen
            if self._transcriber:
                self.callback.on_step("Unloading Transcriber to free VRAM...")
                self._transcriber.unload()
                self._transcriber = None
                torch.cuda.empty_cache()
            
            # 1.2 Visual Analysis (Viral Segments)
            with TransitionContextManager("Visual Analysis Phase"):
                # Stage 1: Analyze video
                self.callback.on_stage_start("Video Analysis", 3)
                analysis = self._run_analysis(str(video_path), transcript_path)
                self.callback.on_stage_complete("Video Analysis", analysis)
            
            # UNLOAD ANALYZER to free VRAM for Editing/Tracking
            if self._analyzer:
                self.callback.on_step("Unloading Qwen3-VL to free VRAM...")
                self._analyzer.unload()
                self._analyzer = None
                torch.cuda.empty_cache()
            
            # Store theme/audience for Gemini enhancement
            self._current_theme = getattr(analysis, 'overall_theme', "") if analysis else ""
            self._current_audience = getattr(analysis, 'target_audience', "") if analysis else ""
            
            if not analysis or not analysis.viral_moments:
                errors.append("No viral moments identified")
                analysis = self._fallback_analysis(str(video_path))
            
            # ================================================================
            # DURATION ENFORCEMENT (Q1+Q6)
            # Qwen returns 20-30s segments — enforce [45, 90]s range
            # Snap boundaries to word boundaries for natural endings
            # ================================================================
            if analysis and analysis.viral_moments:
                # Load transcript words for boundary snapping
                transcript_words = []
                if self.config.transcript_path and os.path.exists(self.config.transcript_path):
                    try:
                        t_data = self._load_transcript(self.config.transcript_path)
                        # Extract flat word list
                        if "result" in t_data and "words" in t_data.get("result", {}):
                            transcript_words = t_data["result"]["words"]
                        elif "words" in t_data:
                            transcript_words = t_data["words"]
                        elif "segments" in t_data:
                            for seg in t_data["segments"]:
                                transcript_words.extend(seg.get("words", []))
                    except Exception as e:
                        print(f"[Orchestrator] Warning: Could not load transcript for duration enforcement: {e}")
                
                self.callback.on_step("Enforcing duration constraints [45-90s]...")
                analysis.viral_moments = self._enforce_duration(
                    analysis.viral_moments,
                    transcript_words,
                    min_duration=self.config.short_duration_range[0],
                    max_duration=self.config.short_duration_range[1]
                )
                print(f"[Orchestrator] After enforcement: {len(analysis.viral_moments)} segments")
                for i, m in enumerate(analysis.viral_moments):
                    dur = m.end_time - m.start_time
                    print(f"  Segment {i+1}: {m.start_time:.1f}s - {m.end_time:.1f}s ({dur:.1f}s)")
            
            # Stage 2: Track subjects
            self.callback.on_stage_start("Subject Tracking", 1)
            tracking_data = self._run_tracking(str(video_path))
            self.callback.on_stage_complete("Subject Tracking", tracking_data)

            # UNLOAD TRACKER
            if self._tracker:
                self.callback.on_step("Unloading Tracker to free VRAM...")
                self._tracker.unload()
                self._tracker = None
            if self._active_speaker:
                self._active_speaker.unload() if hasattr(self._active_speaker, 'unload') else None
                self._active_speaker = None
            torch.cuda.empty_cache()
            
            # Stage 3: Generate EDL (Optional)
            edl = None
            if self.director is not None and transcript_path and not self.config.use_gemini_analysis:
                try:
                    self.callback.on_stage_start("EDL Generation", 1)
                    self.callback.on_step("Using LLM Director to optimize cuts...")
                    
                    edl = self.director.generate_edl(
                        video_path=str(video_path),
                        transcript=self._load_transcript(transcript_path),
                        viral_moments=[
                            m.model_dump() if hasattr(m, 'model_dump') else m.dict() if hasattr(m, 'dict') else asdict(m) 
                            for m in analysis.viral_moments
                        ],
                        num_shorts=self.config.num_shorts,
                        target_duration=self.config.short_duration_range
                    )
                    
                    # Save EDL
                    edl.save(str(output_dir / "edl.json"))
                    self.callback.on_stage_complete("EDL Generation", edl)
                except Exception as e:
                    self.callback.on_error("EDL Generation", str(e))
            
            # Stage 4: Generate shorts
            moments_to_process = analysis.viral_moments[:self.config.num_shorts]
            
            # Use EDL decisions if available
            if edl:
                moments_to_process = []
                for i, decision in enumerate(edl.decisions):
                    # Convert decision back to moment-like object or use enriched data
                    # For now we'll stick to the original moment structure but update times
                    original_moment = analysis.viral_moments[i]
                    original_moment.start_time = decision.source_start
                    original_moment.end_time = decision.source_end
                    original_moment.hook = f"{original_moment.hook} (Optimized)"
                    moments_to_process.append(original_moment)
            
            self.callback.on_stage_start("Short Generation", len(moments_to_process))
            
            for i, moment in enumerate(moments_to_process):
                self.callback.on_step(f"Creating short {i+1}: {moment.hook[:50]}...", i+1, self.config.num_shorts)
                
                try:
                    short_result = self._create_short(
                        source_path=str(video_path),
                        moment=moment,
                        tracking_data=tracking_data,
                        output_dir=output_dir,
                        index=i+1
                    )
                    shorts_created.append(short_result)
                except Exception as e:
                    errors.append(f"Short {i+1} failed: {str(e)}")
            
            self.callback.on_stage_complete("Short Generation", shorts_created)
            
            # Stage 4: Generate metadata
            self.callback.on_stage_start("Metadata Generation", 1)
            self._generate_metadata(output_dir, analysis, shorts_created)
            self.callback.on_stage_complete("Metadata Generation")
            
            # Stage 5: Thumbnails (F2.4)
            if ThumbnailExtractor is not None:
                self.callback.on_stage_start("Thumbnail Extraction", len(shorts_created))
                thumb = ThumbnailExtractor()
                for i, short_meta in enumerate(shorts_created):
                    short_dir = Path(short_meta.get("output_path", "")).parent
                    if short_dir.exists():
                        ts = short_meta.get("thumbnail_timestamp")
                        thumb.extract_for_short(str(short_dir))
                        self.callback.on_step(f"Thumbnail {i+1}/{len(shorts_created)}")
                self.callback.on_stage_complete("Thumbnail Extraction")
            
            # Stage 6: Quality Assurance (G1-G2)
            if QualityAssurance is not None:
                self.callback.on_stage_start("Quality Assurance", 1)
                qa = QualityAssurance()
                qa_reports = qa.check_batch(str(output_dir))
                qa_passed = all(r.overall_passed for r in qa_reports)
                for r in qa_reports:
                    short_dir = Path(r.video_path).parent
                    qa.save_report(r, str(short_dir / "qa_report.json"))
                self.callback.on_step(f"QA: {'All passed' if qa_passed else 'Issues found'}")
                self.callback.on_stage_complete("Quality Assurance")
            
        except Exception as e:
            import traceback
            errors.append(f"Pipeline error: {str(e)}")
            traceback.print_exc()
        finally:
            self._cleanup()
        
        duration = time.time() - start_time
        
        return PipelineResult(
            success=len(shorts_created) > 0,
            video_path=str(video_path),
            output_dir=str(output_dir),
            shorts_created=shorts_created,
            duration_seconds=duration,
            errors=errors,
            metadata={
                "config": asdict(self.config),
                "analysis_segments": len(analysis.viral_moments) if analysis else 0
            }
        )
    
    def _run_analysis(self, video_path: str, transcript_path: Optional[str]):
        """Run video analysis with Gemini first, fallback to Qwen3-VL"""
        
        # 1. Try Gemini Analysis first if enabled (Better transcript understanding)
        if self.config.use_gemini_analysis and transcript_path:
            try:
                self.callback.on_step("Loading Gemini 3 Flash for script analysis (Multi-Cut)...")
                from autonomous_trend_agent.brain.gemini_transcript_analyzer import GeminiTranscriptAnalyzer
                
                gemini_analyzer = GeminiTranscriptAnalyzer()
                
                # Load transcript dictionary
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_dict = json.load(f)
                    
                analysis = gemini_analyzer.analyze_transcript(
                    transcript=transcript_dict,
                    num_shorts=self.config.num_shorts,
                    target_duration=self.config.short_duration_range
                )
                
                if analysis and analysis.shorts:
                    self.callback.on_step(f"Found {len(analysis.shorts)} viral moments (Gemini Multi-Cut)")
                    
                    # Convert TranscriptAnalysis to standard VideoAnalysis
                    from autonomous_trend_agent.brain.video_analyzer import VideoAnalysis, ViralMoment
                    
                    moments = []
                    for short in analysis.shorts:
                        segments_dicts = [{"start": seg.start, "end": seg.end} for seg in short.segments]
                        moments.append(ViralMoment(
                            hook=short.hook,
                            reason=short.reason,
                            viral_score=short.viral_score,
                            segments=segments_dicts
                        ))
                    
                    return VideoAnalysis(
                        video_path=video_path,
                        duration=0.0, # Will be filled if needed later
                        viral_moments=moments,
                        overall_theme=analysis.overall_theme,
                        target_audience=analysis.target_audience
                    )
                else:
                    self.callback.on_step("Gemini returned no moments, falling back...")
            except Exception as e:
                self.callback.on_error("Gemini Analysis", str(e))
                self.callback.on_step("Falling back to Qwen3-VL analysis...")

        # 2. Fallback to Qwen3-VL
        if self.analyzer is None:
            self.callback.on_step("Qwen3-VL disabled, using fallback analysis")
            return self._fallback_analysis(video_path)
        
        try:
            self.callback.on_step("Loading Qwen3-VL model...")
            analysis = self.analyzer.analyze_video(
                video_path,
                transcript_json=transcript_path,
                num_clips=self.config.num_shorts,
                num_frames=32  # More frames for better analysis of long videos
            )
            self.callback.on_step(f"Found {len(analysis.viral_moments)} viral moments")
            return analysis
        except Exception as e:
            self.callback.on_error("Analysis", str(e))
            return self._fallback_analysis(video_path)
    
    def _fallback_analysis(self, video_path: str):
        """Smart Fallback: Audio-Energy based segmentation"""
        self.callback.on_step("Switching to SMART FALLBACK (Audio Energy)...")
        from autonomous_trend_agent.brain.qwen3_video_analyzer import VideoAnalysis, ViralMoment
        
        try:
            import librosa
            import numpy as np
            
            # 1. Load Audio (Resample to 16k for speed)
            # Note: This might take a moment but is faster than VLM
            y, sr = librosa.load(video_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 2. Compute RMS Energy
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            
            # 3. Find peaks (Greedy window selection)
            target_duration = 60.0  # Aim for 60s
            min_duration = 45.0
            
            moments = []
            window_size = int(target_duration * sr / hop_length)
            if window_size > len(rms):
                window_size = len(rms) // 2
            
            # Smooth energy to find sustained high-energy zones
            energy_envelope = np.convolve(rms, np.ones(window_size)/window_size, mode='same')
            
            # Sort indices by energy
            sorted_indices = np.argsort(energy_envelope)[::-1]
            
            mask = np.zeros_like(energy_envelope, dtype=bool)
            frames_per_sec = sr / hop_length
            mask_radius = int(target_duration * frames_per_sec) # Don't overlap
            
            found_count = 0
            for idx in sorted_indices:
                if found_count >= self.config.num_shorts:
                    break
                    
                if mask[idx]:
                    continue
                    
                # Found a peak, define window around it
                center_time = times[idx]
                start_time = max(0.0, center_time - target_duration/2)
                end_time = min(duration, start_time + target_duration)
                
                # Adjust if cut off
                if end_time - start_time < min_duration:
                    continue
                    
                moments.append(ViralMoment(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    hook=f"High Energy Segment {found_count+1}",
                    reason="Detected audio energy spike (Smart Fallback)",
                    viral_score=float(energy_envelope[idx] / (np.max(energy_envelope) + 1e-6))
                ))
                
                # Mask region
                start_mask = max(0, idx - mask_radius)
                end_mask = min(len(mask), idx + mask_radius)
                mask[start_mask:end_mask] = True
                found_count += 1
            
            # Sort chronologically
            moments.sort(key=lambda x: x.start_time)
            
            return VideoAnalysis(
                video_path=video_path,
                duration=duration,
                viral_moments=moments,
                overall_theme="High Energy Highlights",
                target_audience="General"
            )

        except Exception as e:
            self.callback.on_error("Smart Fallback", str(e))
            # Ultimate Fallback (Legacy)
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps if fps > 0 else 60
            cap.release()
            
            segment_duration = min(45, duration / self.config.num_shorts)
            moments = []
            
            for i in range(self.config.num_shorts):
                start = i * segment_duration
                end = min(start + segment_duration, duration)
                moments.append(ViralMoment(
                    start_time=start,
                    end_time=end,
                    hook=f"Segment {i+1}",
                    reason="System Failure Fallback",
                    viral_score=0.5
                ))
            
            return VideoAnalysis(
                video_path=video_path,
                duration=duration,
                viral_moments=moments,
                overall_theme="Unknown",
                target_audience="General"
            )
    
    def _run_tracking(self, video_path: str) -> Dict:
        """Run object tracking with YOLOv11 and optional active speaker detection"""
        result = {"tracked_objects": []}
        
        # Run object tracking
        if self.tracker is not None:
            try:
                self.callback.on_step("Running YOLOv12-Face tracking with ByteTrack...")
                result = self.tracker.track_video(
                    video_path,
                    target_class="face",
                    smooth=True
                )
                
                if result["tracked_objects"]:
                    frames = result["tracked_objects"][0].get("frames_tracked", 0)
                    self.callback.on_step(f"Tracked subject in {frames} frames")
            except Exception as e:
                self.callback.on_error("Tracking", str(e))
        else:
            self.callback.on_step("YOLOv11 disabled, using default center tracking")
        
        # Run active speaker detection
        if self.active_speaker is not None:
            try:
                self.callback.on_step("Detecting active speaker...")
                speaker_result = self.active_speaker.analyze(
                    video_path,
                    face_data=result if result["tracked_objects"] else None
                )
                
                # Add speaker info to result
                result["active_speaker"] = {
                    "num_speakers": speaker_result.num_speakers,
                    "dominant_speaker_id": speaker_result.dominant_speaker_id,
                    "speaker_segments": [
                        {
                            "speaker_id": s.speaker_id,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                            "confidence": s.confidence,
                            "face_bbox": s.face_bbox
                        }
                        for s in speaker_result.speakers
                    ]
                }
                
                self.callback.on_step(f"Detected {speaker_result.num_speakers} speaker(s)")
            except Exception as e:
                self.callback.on_error("Active Speaker", str(e))
        
        return result
    
    def _create_short(
        self,
        source_path: str,
        moment,
        tracking_data: Dict,
        output_dir: Path,
        index: int
    ) -> Dict:
        """Create a single short by stitching together multiple viral segments"""
        import subprocess
        
        # Create short directory
        safe_hook = "".join(c for c in moment.hook[:20] if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        short_name = f"short_{index:02d}_{safe_hook}"
        short_dir = output_dir / short_name
        short_dir.mkdir(exist_ok=True)
        
        output_path = short_dir / "video.mp4"
        
        # If segments list isn't present or is empty, fallback to the single continuous clip logic
        segments_to_process = getattr(moment, 'segments', [])
        if not segments_to_process:
             segments_to_process = [{"start": moment.start_time, "end": moment.end_time}]
             
        reframed_parts = []
        full_transcript = {"words": []}
        current_global_time = 0.0
        
        from autonomous_trend_agent.editor.zero_copy_pipeline import ZeroCopyPipeline
        
        # Loop over every requested cut for this short
        for part_idx, segment_range in enumerate(segments_to_process):
            seg_start = segment_range.get("start", 0.0)
            seg_end = segment_range.get("end", 0.0)
            
            self.callback.on_step(f"[Short {index}] Processing Part {part_idx+1}/{len(segments_to_process)}: {seg_start:.1f}s to {seg_end:.1f}s")
            
            segment_path = short_dir / f"segment_raw_{part_idx}.mp4"
            reframed_path = short_dir / f"part_{part_idx}_reframed.mp4"
            
            # 1. Extract raw segment
            extract_cmd = [
                'ffmpeg', '-y',
                '-ss', str(seg_start),
                '-i', source_path,
                '-t', str(seg_end - seg_start),
                '-filter_complex', '[0:v]setpts=PTS-STARTPTS[v];[0:a]asetpts=PTS-STARTPTS[a]',
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'h264_nvenc',
                '-preset', 'p5',
                '-pix_fmt', 'yuv420p',
                '-b:v', '12M',
                '-maxrate', '15M',
                '-profile:v', 'high',
                '-r', '30',
                '-c:a', 'aac',
                str(segment_path)
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True)
            
            # 2. Filter tracking
            segment_tracking = self._filter_tracking_for_segment(
                tracking_data, 
                seg_start, 
                seg_end
            )
            
            # 3. Extract Transcript for this segment
            segment_transcript = None
            if self.caption_engine is not None and self.config.transcript_path:
                 try:
                    segment_transcript = self._extract_segment_transcript(
                        self.config.transcript_path,
                        seg_start,
                        seg_end
                    )
                 except Exception as e:
                    print(f"[Orchestrator] Warning: Failed to extract transcript for segment part {part_idx}: {e}")

            # 4. Reframe with hardware
            used_gpu_captions = False
            if isinstance(self.reframer, ZeroCopyPipeline):
                success = self.reframer.reframe_with_tracking(
                    str(segment_path),
                    segment_tracking,
                    str(reframed_path),
                    transcript=segment_transcript,
                    caption_engine=self.caption_engine if self.config.use_captions else None,
                    pacing_plan=[],
                    effects_plan=None,
                    progress_callback=lambda c, t: None
                )
                if not success:
                    raise RuntimeError(f"ZeroCopyPipeline failed to render {segment_path}")
                used_gpu_captions = True
            elif self.config.use_gpu_reframing and hasattr(self.reframer, 'use_gpu') and self.reframer.use_gpu:
                self.reframer.reframe_gpu(str(segment_path), segment_tracking, str(reframed_path))
            else:
                self.reframer.reframe(str(segment_path), segment_tracking, str(reframed_path))
            
            # 5. Burn CPU captions if GPU didn't do it
            final_part_path = reframed_path
            if self.caption_engine is not None and segment_transcript and not used_gpu_captions:
                try:
                    if segment_transcript.get("words"):
                        captioned_path = short_dir / f"part_{part_idx}_captioned.mp4"
                        self.caption_engine.add_captions(
                            str(reframed_path),
                            segment_transcript,
                            str(captioned_path),
                            use_nvenc=True
                        )
                        reframed_path.unlink(missing_ok=True)
                        final_part_path = captioned_path
                except Exception as e:
                    print(f"Caption burn failed for part {part_idx}: {e}")
            
            reframed_parts.append(final_part_path)
            segment_path.unlink(missing_ok=True) # Cleanup raw extraction
            
            # 6. Accumulate flattened transcript for Metadata saving
            if segment_transcript and segment_transcript.get("words"):
                for w in segment_transcript["words"]:
                    adjusted_word = w.copy()
                    adjusted_word["start_time"] = round(w["start_time"] + current_global_time, 3)
                    adjusted_word["end_time"] = round(w["end_time"] + current_global_time, 3)
                    full_transcript["words"].append(adjusted_word)
            current_global_time += (seg_end - seg_start)
            
        # ========================================================
        # STITCHING: Concatenate all processed parts
        # ========================================================
        self.callback.on_step(f"[Short {index}] Stitching {len(reframed_parts)} reframed parts together")
        
        concat_list_path = short_dir / "concat_list.txt"
        with open(concat_list_path, 'w') as f:
            for part in reframed_parts:
                f.write(f"file '{part.name}'\n")
                
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list_path),
            '-c', 'copy',  # Zero-Copy concatenation!
            str(output_path)
        ]
        
        subprocess.run(concat_cmd, capture_output=True, check=True)
        
        # Cleanup intermediary parts
        concat_list_path.unlink(missing_ok=True)
        for part in reframed_parts:
            part.unlink(missing_ok=True)
            
        # Apply global effects across the final stitched short
        if self.effects is not None:
            try:
                plan = self.effects.analyze_and_plan(str(output_path))
                self.effects.audio_analyzer.save_plan(plan, str(short_dir / "effects_plan.json"))
            except Exception as e:
                print(f"Effects failed: {e}")
        
        # === GEMINI AI ENHANCEMENT ===
        ai_enhancement = None
        if self.gemini_services is not None:
            try:
                self.callback.on_step(f"[Short {index}] Running AI enhancement...")
                hook_text = moment.hook if hasattr(moment, 'hook') else ""
                
                ai_enhancement = self.gemini_services.enhance_short(
                    transcript_text=hook_text,
                    hook_text=hook_text,
                    theme=getattr(self, '_current_theme', ""),
                    audience=getattr(self, '_current_audience', ""),
                    duration_seconds=current_global_time
                )
                self.gemini_services.save_enhancement(ai_enhancement, str(short_dir / "ai_enhancement.json"))
                self.callback.on_step(f"[Short {index}] AI: Hook {ai_enhancement.hook_score.score}/100, QC {'✓' if ai_enhancement.qc_result.passed else '✗'}")
            except Exception as e:
                print(f"[Short {index}] AI enhancement failed: {e}")
        
        # Save short metadata (with AI enhancements if available)
        short_meta = {
            "index": index,
            "source_start": moment.start_time,
            "source_end": moment.end_time,
            "hook": moment.hook,
            "viral_score": moment.viral_score,
            "output_path": str(output_path)
        }
        
        # Merge AI-generated metadata if available
        if ai_enhancement is not None:
            short_meta["title"] = ai_enhancement.metadata.title
            short_meta["description"] = ai_enhancement.metadata.description
            short_meta["tags"] = ai_enhancement.metadata.tags
            short_meta["hashtags"] = ai_enhancement.metadata.hashtags
            short_meta["hook_score"] = ai_enhancement.hook_score.score
            short_meta["qc_passed"] = ai_enhancement.qc_result.passed
            short_meta["caption_style"] = ai_enhancement.caption_style
            short_meta["thumbnail_timestamp"] = ai_enhancement.thumbnail.timestamp
        
        with open(short_dir / "metadata.json", "w") as f:
            json.dump(short_meta, f, indent=2)
        
        # Save transcript.txt for this short (F3.3)
        if segment_transcript and segment_transcript.get("words"):
            transcript_text = " ".join(
                w.get("word", w.get("text", "")) 
                for w in segment_transcript["words"]
            )
            with open(short_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript_text)
            # Also save structured transcript
            with open(short_dir / "transcript.json", "w") as f:
                json.dump(segment_transcript, f, indent=2)
        
        return short_meta
    
    def _load_transcript(self, transcript_path: str) -> Dict:
        """Load transcript JSON"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_segment_transcript(
        self,
        transcript_path: str,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        Extract words for a specific segment from transcript.
        Implements 'TranscriptAligner' logic:
        - Handle boundary straddling (words starting before cut)
        - Clamp to valid clip duration
        """
        import json
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        words = []
        source_words = []
        
        # Parakeet format
        if "result" in transcript and "words" in transcript.get("result", {}):
            source_words = transcript["result"]["words"]
        # Whisper format
        elif "segments" in transcript:
            for seg in transcript["segments"]:
                source_words.extend(seg.get("words", []))
        # Flat words list
        elif "words" in transcript:
            source_words = transcript["words"]
        
        clip_duration = end_time - start_time
        
        for w in source_words:
            w_start = w.get("start_time", w.get("start", 0))
            w_end = w.get("end_time", w.get("end", 0))
            text = w.get("word", w.get("text", ""))
            
            # 1. Reject segments fully outside
            if w_end < start_time:
                continue
            if w_start > end_time:
                break # Assuming sorted
            
            # 2. Calculate relative timestamps
            # T_new = T_old - Cut_Start
            new_start = w_start - start_time
            new_end = w_end - start_time
            
            # 3. Handle Boundary Straddling (Clamping)
            # If word starts before 0 (straddle), clamp to 0 so it appears immediately
            new_start = max(0.0, new_start)
            # If word ends after clip, clamp to duration
            new_end = min(clip_duration, new_end)
            
            # 4. Filter micro-segments (validation)
            if new_end - new_start > 0.1:
                words.append({
                    "word": text,
                    "start_time": round(new_start, 3),
                    "end_time": round(new_end, 3)
                })
                
        return {"words": words}
    
    def _filter_tracking_for_segment(
        self, 
        tracking_data: Dict, 
        start_time: float, 
        end_time: float
    ) -> Dict:
        """Filter tracking data and dynamically stitch active speaker trajectories"""
        if not tracking_data.get("tracked_objects"):
            return {"tracked_objects": []}
        
        fps = tracking_data.get("fps", 30)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # 1. Map frame_idx -> available tracking points for ALL identities
        frame_to_tracks = {}
        for obj in tracking_data["tracked_objects"]:
            track_id = obj.get("id")
            for pt in obj.get("trajectory", []):
                f_idx = pt.get("frame_idx", 0)
                if start_frame <= f_idx <= end_frame:
                    if f_idx not in frame_to_tracks:
                        frame_to_tracks[f_idx] = {}
                    frame_to_tracks[f_idx][track_id] = pt
        
        # 2. Map frame_idx -> active speaker ID
        active_speaker_map = {}
        active_speaker_data = tracking_data.get("active_speaker", {})
        dom_speaker = active_speaker_data.get("dominant_speaker_id")
        
        for seg in active_speaker_data.get("speaker_segments", []):
            s_start = int(seg["start_time"] * fps)
            s_end = int(seg["end_time"] * fps)
            for f in range(s_start, s_end + 1):
                active_speaker_map[f] = seg["speaker_id"]
        
        # 3. Compile the Director's Synthesized Trajectory
        synth_trajectory = []
        last_valid_id = None
        
        for f_idx in sorted(frame_to_tracks.keys()):
            available_tracks = frame_to_tracks[f_idx]
            
            # Determine who we should be looking at right now
            target_id = active_speaker_map.get(f_idx)
            
            # Fallbacks if the target isn't visible in this specific frame
            if target_id not in available_tracks:
                if last_valid_id in available_tracks:
                    target_id = last_valid_id  # Keep looking at previous person
                elif dom_speaker in available_tracks:
                    target_id = dom_speaker    # Look at main subject
                else:
                    # Look at whoever is available (largest track logic essentially)
                    target_id = list(available_tracks.keys())[0]
            
            pt = available_tracks[target_id]
            
            # Normalize frame index for the cut segment chunk
            new_pt = pt.copy()
            new_pt["frame_idx"] = f_idx - start_frame
            
            synth_trajectory.append(new_pt)
            last_valid_id = target_id
            
        if not synth_trajectory:
            return {"tracked_objects": []}

        # Return a single stitched trajectory exactly as `ZeroCopyPipeline` expects
        return {
            "fps": fps,
            "tracked_objects": [{
                "label": "person",
                "id": "director_cut",
                "frames_tracked": len(synth_trajectory),
                "trajectory": synth_trajectory
            }]
        }
    
    # =========================================================================
    # Q1 + Q6: Duration Enforcement + Natural Endings
    # =========================================================================
    def _enforce_duration(
        self,
        moments: List,
        transcript_words: List[Dict],
        min_duration: int = 55,
        max_duration: int = 90
    ) -> List:
        """
        Enforce duration constraints on viral moments.
        
        Strategy (from research: Contextual Adjacent Segment Merging):
        1. Reject segments < 30s (unsalvageable)
        2. Extend 30-44s segments forward using word-boundary snapping
        3. Cap > 90s segments by snapping back to nearest sentence boundary
        4. Validate all segments are in [min_duration, max_duration]
        """
        enforced = []
        
        for moment in moments:
            if not moment.segments:
                continue
                
            # 0. Snap every raw LLM segment to real word/sentence boundaries to prevent internal abrupt cuts
            for seg in moment.segments:
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                
                # Snap start forward to start of next clean sentence/word
                new_start = self._snap_to_word_boundary(seg_start, transcript_words, direction='forward')
                if new_start is not None:
                    seg["start"] = new_start
                    
                # Snap end backward to end of previous clean sentence/word
                new_end = self._snap_to_word_boundary(seg_end, transcript_words, direction='backward')
                if new_end is not None and new_end > seg["start"]:
                    seg["end"] = new_end
                    
            # Calculate total stitched duration
            duration = sum(s.get("end", 0.0) - s.get("start", 0.0) for s in moment.segments)
            
            # 1. Reject unsalvageable short segments
            if duration < 30:
                print(f"[Duration] REJECTED: {duration:.1f}s segment '{moment.hook[:30]}' (< 30s minimum)")
                continue
                
            first_seg = moment.segments[0]
            last_seg = moment.segments[-1]
            
            # 2. Extend short segments (30-44s) toward min_duration
            if duration < min_duration:
                needed = min_duration - duration
                print(f"[Duration] Extending {duration:.1f}s segment by ~{needed:.1f}s...")
                
                # Try to extend forward first (research: forward content more likely continues topic)
                target_end = last_seg["end"] + needed + 5  # Add 5s buffer for boundary snapping
                snapped_end = self._snap_to_word_boundary(
                    target_end, transcript_words, direction='backward'
                )
                
                if snapped_end is not None and snapped_end > last_seg["end"]:
                    last_seg["end"] = snapped_end
                else:
                    # Fallback: extend backward
                    target_start = first_seg["start"] - needed - 5
                    snapped_start = self._snap_to_word_boundary(
                        max(0, target_start), transcript_words, direction='forward'
                    )
                    if snapped_start is not None and snapped_start < first_seg["start"]:
                        first_seg["start"] = snapped_start
                    else:
                        # Last resort: blind extend forward
                        last_seg["end"] += needed
                
                duration = sum(s.get("end", 0.0) - s.get("start", 0.0) for s in moment.segments)
                print(f"[Duration] Extended to {duration:.1f}s")
            
            # 3. Cap segments exceeding max_duration
            if duration > max_duration:
                extra = duration - max_duration
                print(f"[Duration] Capping {duration:.1f}s segment to ~{max_duration}s...")
                target_end = last_seg["end"] - extra
                snapped_end = self._snap_to_word_boundary(
                    target_end, transcript_words, direction='backward'
                )
                if snapped_end is not None and snapped_end > first_seg["start"] + 30:
                    last_seg["end"] = snapped_end
                else:
                    last_seg["end"] = target_end
                duration = sum(s.get("end", 0.0) - s.get("start", 0.0) for s in moment.segments)
                print(f"[Duration] Capped to {duration:.1f}s")
            
            # 4. Snap start to word boundary too (natural beginning)
            snapped_start = self._snap_to_word_boundary(
                first_seg["start"], transcript_words, direction='forward'
            )
            if snapped_start is not None:
                # Only use if it doesn't reduce duration too much
                if duration - (snapped_start - first_seg["start"]) >= min_duration * 0.9:
                    first_seg["start"] = snapped_start
                    duration = sum(s.get("end", 0.0) - s.get("start", 0.0) for s in moment.segments)
            
            # 5. Final validation
            if duration >= min_duration * 0.9:  # Allow 10% tolerance
                enforced.append(moment)
                print(f"[Duration] ACCEPTED: {duration:.1f}s segment '{moment.hook[:30]}'")
            else:
                print(f"[Duration] REJECTED: {duration:.1f}s segment '{moment.hook[:30]}' (after enforcement < min)")
                
        return enforced
    
    def _snap_to_word_boundary(
        self,
        target_time: float,
        transcript_words: List[Dict],
        direction: str = 'backward',
        search_window: float = 8.0
    ) -> Optional[float]:
        """
        Find the safest cut point near target_time using multi-modal boundary detection.
        
        Boundary quality (highest to lowest priority):
        1. Word followed by >300ms silence AND sentence-ending punctuation (. ? !)
        2. Word followed by sentence-ending punctuation
        3. Word followed by >300ms silence
        4. Nearest complete word boundary
        
        Args:
            target_time: Target cut time in seconds
            transcript_words: Flat list of word dicts with start_time/end_time
            direction: 'forward' = search after target, 'backward' = search before target
            search_window: How far (seconds) to search from target
            
        Returns:
            Best cut time (end_time of the chosen word), or None if no words found
        """
        if not transcript_words:
            return None
        
        # Standardize key names
        def get_start(w):
            return float(w.get("start_time", w.get("start", 0)))
        def get_end(w):
            return float(w.get("end_time", w.get("end", 0)))
        def get_text(w):
            return w.get("word", w.get("text", ""))
        
        # Filter words within search window
        window_start = target_time - search_window
        window_end = target_time + search_window
        
        candidates = []
        for i, w in enumerate(transcript_words):
            w_end = get_end(w)
            if window_start <= w_end <= window_end:
                text = get_text(w).strip()
                
                # Calculate silence gap after this word
                silence_after = 0.0
                if i + 1 < len(transcript_words):
                    next_start = get_start(transcript_words[i + 1])
                    silence_after = next_start - w_end
                else:
                    silence_after = 999.0  # Last word — infinite silence
                
                # Score this boundary
                is_sentence_end = text.rstrip().endswith(('.', '?', '!', '...', '"'))
                has_long_silence = silence_after > 0.3  # 300ms threshold
                
                # Multi-modal score: higher = better boundary
                score = 0
                if is_sentence_end and has_long_silence:
                    score = 4  # Best: punctuation + acoustic pause
                elif is_sentence_end:
                    score = 3  # Good: sentence end
                elif has_long_silence:
                    score = 2  # Okay: natural pause
                else:
                    score = 1  # Fallback: word boundary
                
                # Proximity penalty (prefer closer to target)
                proximity = abs(w_end - target_time)
                
                candidates.append({
                    'time': w_end,
                    'score': score,
                    'proximity': proximity,
                    'text': text,
                    'silence': silence_after
                })
        
        if not candidates:
            return None
        
        # Sort: highest score first, then smallest proximity
        candidates.sort(key=lambda c: (-c['score'], c['proximity']))
        
        best = candidates[0]
        if best['score'] >= 2:
            print(f"[WordBoundary] Snapped to '{best['text']}' at {best['time']:.3f}s "
                  f"(score={best['score']}, silence={best['silence']:.3f}s)")
        
        return best['time']

    def _generate_metadata(self, output_dir: Path, analysis, shorts: List[Dict]):
        """Generate batch metadata"""
        batch_meta = {
            "generated_at": datetime.now().isoformat(),
            "source_video": self.config.video_path,
            "total_shorts": len(shorts),
            "theme": analysis.overall_theme if analysis else "Unknown",
            "target_audience": analysis.target_audience if analysis else "General",
            "shorts": shorts,
            "config": asdict(self.config)
        }
        
        with open(output_dir / "batch_metadata.json", "w") as f:
            json.dump(batch_meta, f, indent=2)
    
    def _cleanup(self):
        """Cleanup loaded models to free VRAM"""
        if self._analyzer is not None:
            try:
                self._analyzer.unload()
            except:
                pass
        
        if self._tracker is not None:
            try:
                self._tracker.unload()
            except:
                pass
        
        torch.cuda.empty_cache()

    # Alias for backward compatibility
    def process_video(self, video_path: str, transcript_path: Optional[str] = None) -> PipelineResult:
        return self.run(video_path, transcript_path)


def run_pipeline(
    video_path: str,
    transcript_path: Optional[str] = None,
    output_dir: str = "./output",
    num_shorts: int = 4,
    use_gpu: bool = True
) -> PipelineResult:
    """
    Convenience function to run the full pipeline.
    
    Args:
        video_path: Path to source video
        transcript_path: Optional transcript JSON
        output_dir: Output directory
        num_shorts: Number of shorts to generate
        use_gpu: Use GPU acceleration
        
    Returns:
        PipelineResult with generated shorts
    """
    # Check for GPU reframing env var (default OFF due to memory issues)
    use_gpu_reframing = os.getenv("USE_GPU_REFRAMING", "0") == "1"
    
    config = PipelineConfig(
        output_dir=output_dir,
        num_shorts=num_shorts,
        use_gpu_reframing=use_gpu_reframing,
        use_yolo_tracking=use_gpu,
        use_qwen_analysis=os.getenv("SKIP_ANALYSIS") != "1",  # Disable if env var set
        device="cuda" if use_gpu else "cpu"
    )
    
    orchestrator = PipelineOrchestrator(config)
    try:
        return orchestrator.process_video(video_path, transcript_path)
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    import sys
    import shutil
    import torch
    
    # If no video path is provided, default to the downloaded_videos directory
    video_path_arg = sys.argv[1] if len(sys.argv) > 1 else "autonomous_trend_agent/downloaded_videos"
    transcript = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else "./output"
    
    print("\n" + "="*60)
    print("AUTONOMOUS TREND AGENT - GOLDEN STACK PIPELINE")
    print("="*60 + "\n")
    
    # Auto-detect GPU
    use_gpu = torch.cuda.is_available()
    print(f"Stats: GPU Available: {use_gpu}")
    
    video_path_obj = Path(video_path_arg)
    
    if video_path_obj.is_dir():
        # ========== BATCH MODE ==========
        project_root = video_path_obj
        completed_dir = project_root / "completed_videos"
        failed_dir = project_root / "failed_videos"
        
        completed_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather all typical video formats
        videos_to_process = []
        for ext in ["*.mp4", "*.mkv", "*.webm", "*.mov"]:
            videos_to_process.extend(video_path_obj.glob(ext))
            
        if not videos_to_process:
            print(f"No videos found to process in {video_path_obj.absolute()}")
            sys.exit(0)
            
        print(f"Found {len(videos_to_process)} videos to process in {video_path_obj.absolute()}")
        
        success_count = 0
        for vid in videos_to_process:
            print(f"\n{'*'*40}")
            print(f"Processing: {vid.name}")
            print(f"{'*'*40}")
            try:
                result = run_pipeline(str(vid), transcript, output, use_gpu=use_gpu)
                if result.success:
                    success_count += 1
                    dest = completed_dir / vid.name
                    if dest.exists(): dest.unlink()
                    shutil.move(str(vid), str(dest))
                    print(f"→ Moved {vid.name} to {completed_dir}")
                else:
                    dest = failed_dir / vid.name
                    if dest.exists(): dest.unlink()
                    shutil.move(str(vid), str(dest))
                    print(f"→ Moved {vid.name} to {failed_dir} (Pipeline returned Failure)")
            except Exception as e:
                print(f"Unhandled exact error processing {vid.name}: {e}")
                dest = failed_dir / vid.name
                if dest.exists(): dest.unlink()
                shutil.move(str(vid), str(dest))
                print(f"→ Moved {vid.name} to {failed_dir} (Unhandled Exception)")

        print("\n" + "="*60)
        print("BATCH PIPELINE COMPLETE")
        print("="*60)
        print(f"Total Processed: {len(videos_to_process)}")
        print(f"Successful:      {success_count}")
        print(f"Failed:          {len(videos_to_process) - success_count}")
        
        sys.exit(0 if success_count == len(videos_to_process) else 1)

    else:
        # ========== SINGLE FILE MODE ==========
        if not video_path_obj.exists():
            print(f"Error: Target file {video_path_obj.absolute()} does not exist.")
            sys.exit(1)
            
        result = run_pipeline(str(video_path_obj), transcript, output, use_gpu=use_gpu)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Success: {result.success}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Shorts Created: {len(result.shorts_created)}")
        print(f"Output: {result.output_dir}")
        
        if result.errors:
            print(f"\nErrors:")
            for e in result.errors:
                print(f"  - {e}")
        
        print("")
        sys.exit(0 if result.success else 1)

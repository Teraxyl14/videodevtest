"""
Context-Aware Effects Engine
Applies dynamic zoom, shake, and emphasis effects based on audio energy
Uses librosa for audio analysis and Kornia for GPU-accelerated effects
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import scipy.signal
# Fix for newer scipy versions where hann was moved
if not hasattr(scipy.signal, 'hann'):
    import scipy.signal.windows
    scipy.signal.hann = scipy.signal.windows.hann


@dataclass
class EffectTrigger:
    """A point in the video where an effect should be applied"""
    timestamp: float      # In seconds
    effect_type: str      # 'zoom', 'shake', 'flash', 'emphasize'
    intensity: float      # 0.0 to 1.0
    duration: float       # Effect duration in seconds
    metadata: Optional[Dict] = None


@dataclass
class EffectPlan:
    """Complete effect plan for a video segment"""
    duration: float
    triggers: List[EffectTrigger]
    peak_timestamps: List[float]
    average_energy: float


class AudioAnalyzer:
    """Analyze audio to detect emphasis points for effects"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize audio analyzer.
        
        Note: librosa runs on CPU, but we can precompute effect parameters
        for GPU application.
        """
        self.device = device
        self._librosa = None
    
    def _ensure_loaded(self):
        """Lazy import librosa"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
    
    def analyze_audio(
        self,
        audio_path: str,
        sr: int = 22050,
        hop_length: int = 512
    ) -> EffectPlan:
        """
        Analyze audio file to find effect trigger points.
        
        Args:
            audio_path: Path to audio file (or video with audio)
            sr: Sample rate for analysis
            hop_length: Hop length for STFT
            
        Returns:
            EffectPlan with trigger points
        """
        self._ensure_loaded()
        librosa = self._librosa
        
        print(f"[AudioAnalyzer] Analyzing: {audio_path}")
        
        # Extract audio to WAV first (avoids PySoundFile warning on video files)
        import subprocess, tempfile, os as _os
        tmp_wav = None
        if not str(audio_path).lower().endswith(('.wav', '.flac', '.ogg')):
            tmp_wav = tempfile.mktemp(suffix='.wav')
            subprocess.run(
                ['ffmpeg', '-y', '-v', 'error', '-i', audio_path, '-vn', '-acodec', 'pcm_s16le',
                 '-ar', str(sr), tmp_wav],
                capture_output=True, check=True
            )
            load_path = tmp_wav
        else:
            load_path = audio_path
        
        y, sr = librosa.load(load_path, sr=sr)
        
        if tmp_wav and _os.path.exists(tmp_wav):
            _os.unlink(tmp_wav)
        duration = len(y) / sr
        
        print(f"[AudioAnalyzer] Duration: {duration:.1f}s, Sample Rate: {sr}")
        
        triggers = []
        
        # 1. RMS Energy Analysis (for zoom/emphasis)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        
        # Normalize RMS
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        
        # Find peaks (high energy moments)
        peaks = []
        threshold = np.percentile(rms_norm, 85)  # Top 15%
        
        in_peak = False
        peak_start = 0
        
        for i, (t, e) in enumerate(zip(rms_times, rms_norm)):
            if e > threshold and not in_peak:
                in_peak = True
                peak_start = t
            elif e < threshold * 0.8 and in_peak:
                in_peak = False
                peak_duration = t - peak_start
                if peak_duration > 0.1:  # Minimum peak duration
                    peaks.append({
                        "timestamp": peak_start,
                        "duration": peak_duration,
                        "intensity": float(rms_norm[i-1]) if i > 0 else float(e)
                    })
        
        # Create zoom triggers from peaks
        for peak in peaks[:20]:  # Limit to 20 peaks
            triggers.append(EffectTrigger(
                timestamp=peak["timestamp"],
                effect_type="zoom",
                intensity=peak["intensity"],
                duration=min(peak["duration"], 1.0),
                metadata={"source": "rms_peak"}
            ))
        
        # 2. Onset Detection (for shake/flash)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length,
            backtrack=True, units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        onset_strengths = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Strong onsets trigger shake
        strength_threshold = np.percentile(onset_strengths, 90)
        
        for i, (frame, time) in enumerate(zip(onset_frames, onset_times)):
            if frame < len(onset_strengths) and onset_strengths[frame] > strength_threshold:
                triggers.append(EffectTrigger(
                    timestamp=float(time),
                    effect_type="shake",
                    intensity=float(onset_strengths[frame] / onset_strengths.max()),
                    duration=0.15,
                    metadata={"source": "onset_strong"}
                ))
        
        # 3. Beat Detection (for rhythmic emphasis)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        
        print(f"[AudioAnalyzer] Detected tempo: {tempo:.1f} BPM, {len(beat_times)} beats")
        
        # Add subtle emphasis on beats (every 4th beat)
        for i, beat_time in enumerate(beat_times):
            if i % 4 == 0:  # Downbeats
                triggers.append(EffectTrigger(
                    timestamp=float(beat_time),
                    effect_type="emphasize",
                    intensity=0.3,
                    duration=0.1,
                    metadata={"source": "downbeat", "beat_idx": i}
                ))
        
        # Sort triggers by timestamp
        triggers.sort(key=lambda t: t.timestamp)
        
        # Deduplicate nearby triggers (within 0.1s)
        deduplicated = []
        for trigger in triggers:
            if not deduplicated or trigger.timestamp - deduplicated[-1].timestamp > 0.1:
                deduplicated.append(trigger)
            elif trigger.intensity > deduplicated[-1].intensity:
                deduplicated[-1] = trigger
        
        print(f"[AudioAnalyzer] Generated {len(deduplicated)} effect triggers")
        
        return EffectPlan(
            duration=duration,
            triggers=deduplicated,
            peak_timestamps=[p["timestamp"] for p in peaks],
            average_energy=float(np.mean(rms))
        )
    
    def save_plan(self, plan: EffectPlan, output_path: str):
        """Save effect plan to JSON"""
        data = {
            "duration": plan.duration,
            "average_energy": plan.average_energy,
            "peak_timestamps": plan.peak_timestamps,
            "triggers": [
                {
                    "timestamp": t.timestamp,
                    "effect_type": t.effect_type,
                    "intensity": t.intensity,
                    "duration": t.duration,
                    "metadata": t.metadata
                }
                for t in plan.triggers
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[AudioAnalyzer] Saved effect plan to: {output_path}")


class GPUEffectsProcessor:
    """Apply visual effects using Kornia (GPU-accelerated)"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._kornia = None
    
    def _ensure_loaded(self):
        """Lazy import kornia"""
        if self._kornia is None:
            try:
                import kornia
                self._kornia = kornia
            except Exception:
                pass
    
    def apply_zoom(
        self,
        frames: torch.Tensor,
        zoom_factor: float = 1.2,
        center: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Apply zoom effect to frames.
        
        Args:
            frames: (N, C, H, W) tensor
            zoom_factor: 1.0 = no zoom, >1.0 = zoom in
            center: Normalized center point (0.5, 0.5) = center
            
        Returns:
            Zoomed frames tensor
        """
        self._ensure_loaded()
        kornia = self._kornia
        
        N, C, H, W = frames.shape
        
        if center is None:
            center = (0.5, 0.5)
        
        # Calculate crop box
        crop_h = int(H / zoom_factor)
        crop_w = int(W / zoom_factor)
        
        crop_y = int((H - crop_h) * center[1])
        crop_x = int((W - crop_w) * center[0])
        
        # Crop and resize back to original size
        cropped = frames[:, :, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        if self._kornia:
             zoomed = self._kornia.geometry.transform.resize(cropped, (H, W), interpolation='bilinear')
        else:
             import torch.nn.functional as F
             zoomed = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        
        return zoomed
    
    def apply_shake(
        self,
        frames: torch.Tensor,
        intensity: float = 0.05,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply camera shake effect with Perlin-like smooth noise.
        
        ENHANCED: Uses smoothed random noise instead of pure random per-frame,
        creating a more cinematic "handheld camera" feel instead of jitter.
        Default intensity boosted to 0.05 (from 0.02) for visible impact.
        """
        self._ensure_loaded()
        
        N, C, H, W = frames.shape
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate smooth noise: random per-frame then low-pass filter
        raw_tx = (torch.rand(N, device=frames.device) - 0.5) * 2 * intensity * W
        raw_ty = (torch.rand(N, device=frames.device) - 0.5) * 2 * intensity * H
        
        # 3-frame moving average for smooth camera shake (not jittery)
        if N >= 3:
            kernel = torch.ones(3, device=frames.device) / 3
            # Pad and convolve
            tx_padded = torch.cat([raw_tx[:1], raw_tx, raw_tx[-1:]])
            ty_padded = torch.cat([raw_ty[:1], raw_ty, raw_ty[-1:]])
            tx = torch.conv1d(tx_padded.view(1, 1, -1), kernel.view(1, 1, -1))[0, 0]
            ty = torch.conv1d(ty_padded.view(1, 1, -1), kernel.view(1, 1, -1))[0, 0]
        else:
            tx, ty = raw_tx, raw_ty
        
        # Build affine matrices
        theta = torch.zeros(N, 2, 3, device=frames.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = tx / (W / 2)
        theta[:, 1, 2] = ty / (H / 2)
        
        grid = torch.nn.functional.affine_grid(theta, frames.size(), align_corners=False)
        shaken = torch.nn.functional.grid_sample(frames, grid, align_corners=False, mode='bilinear')
        
        return shaken
    
    def apply_flash(
        self,
        frames: torch.Tensor,
        intensity: float = 0.3
    ) -> torch.Tensor:
        """
        Apply white flash effect.
        
        Args:
            frames: (N, C, H, W) tensor
            intensity: Flash brightness (0.0 - 1.0)
            
        Returns:
            Flashed frames tensor
        """
        # Simple additive flash
        flash = torch.ones_like(frames) * intensity
        flashed = torch.clamp(frames + flash, 0, 1)
        
        return flashed
    
    def apply_vignette(
        self,
        frames: torch.Tensor,
        strength: float = 0.5
    ) -> torch.Tensor:
        """
        Apply vignette effect (darker edges).
        
        Args:
            frames: (N, C, H, W) tensor
            strength: Vignette darkness (0.0 - 1.0)
            
        Returns:
            Vignetted frames tensor
        """
        N, C, H, W = frames.shape
        
        # Create vignette mask
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frames.device),
            torch.linspace(-1, 1, W, device=frames.device),
            indexing='ij'
        )
        
        dist = torch.sqrt(x**2 + y**2)
        vignette = 1 - (dist * strength).clamp(0, 1)
        vignette = vignette.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return frames * vignette
    
    def process_with_plan(
        self,
        frames: torch.Tensor,
        plan: EffectPlan,
        fps: float,
        batch_size: int = 50
    ) -> torch.Tensor:
        """
        Apply effects to frames according to effect plan.
        
        ENHANCED: Stronger intensities across all effects.
        Added zoom_punch (1.3x elastic bounce) and vignette_pulse.
        """
        N = frames.shape[0]
        result_batches = []
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch = frames[batch_start:batch_end].to(self.device)
            
            for frame_idx in range(batch.shape[0]):
                global_idx = batch_start + frame_idx
                current_time = global_idx / fps
                
                for trigger in plan.triggers:
                    if trigger.timestamp <= current_time <= trigger.timestamp + trigger.duration:
                        progress = (current_time - trigger.timestamp) / trigger.duration
                        
                        # Smooth ease in-out
                        eased = 0.5 - 0.5 * np.cos(progress * np.pi)
                        
                        frame = batch[frame_idx:frame_idx+1]
                        
                        if trigger.effect_type == "zoom":
                            # BOOSTED: 0.25 multiplier (was 0.15)
                            zoom = 1.0 + (0.25 * trigger.intensity * eased)
                            frame = self.apply_zoom(frame, zoom_factor=zoom)
                        
                        elif trigger.effect_type == "zoom_punch":
                            # NEW: Elastic bounce zoom — peaks at 1.3x then settles
                            bounce = np.exp(-3.0 * progress) * np.sin(progress * np.pi * 2)
                            zoom = 1.0 + (0.30 * trigger.intensity * max(0, bounce))
                            frame = self.apply_zoom(frame, zoom_factor=zoom)
                        
                        elif trigger.effect_type == "shake":
                            # BOOSTED: 0.05 base (was 0.02), decays over duration
                            shake_int = trigger.intensity * 0.05 * (1 - eased)
                            frame = self.apply_shake(frame, intensity=shake_int, seed=global_idx)
                        
                        elif trigger.effect_type == "flash":
                            # BOOSTED: 0.35 peak (was 0.2), quick decay
                            flash_int = trigger.intensity * 0.35 * (1 - progress ** 0.5)
                            frame = self.apply_flash(frame, intensity=flash_int)
                        
                        elif trigger.effect_type == "emphasize":
                            # Subtle breathing zoom pulse
                            zoom = 1.0 + (0.08 * trigger.intensity * np.sin(progress * np.pi))
                            frame = self.apply_zoom(frame, zoom_factor=zoom)
                        
                        elif trigger.effect_type == "vignette_pulse":
                            # NEW: Vignette that intensifies on beat
                            pulse = 0.3 + 0.4 * trigger.intensity * (1 - progress)
                            frame = self.apply_vignette(frame, strength=pulse)
                        
                        batch[frame_idx] = frame[0]
            
            result_batches.append(batch.cpu())
        
        return torch.cat(result_batches, dim=0)


class ContextAwareEffectsEngine:
    """
    Main effects engine combining audio analysis and GPU effects.
    """
    
    def __init__(self, device: str = "cuda"):
        self.audio_analyzer = AudioAnalyzer()
        self.effects_processor = GPUEffectsProcessor(device=device)
        self.device = device
    
    def analyze_and_plan(self, video_path: str) -> EffectPlan:
        """Analyze video audio and create effect plan"""
        return self.audio_analyzer.analyze_audio(video_path)
    
    def apply_effects(
        self,
        frames: torch.Tensor,
        plan: EffectPlan,
        fps: float
    ) -> torch.Tensor:
        """Apply effect plan to frames"""
        return self.effects_processor.process_with_plan(frames, plan, fps)
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        apply_vignette: bool = True
    ):
        """
        Full pipeline: analyze audio, apply effects, save video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            apply_vignette: Add cinematic vignette
        """
        from autonomous_trend_agent.editor.gpu_video_utils import (
            decode_video_batched,
            encode_tensor_to_video,
            get_video_info,
            clear_gpu_cache
        )
        
        print(f"[EffectsEngine] Processing: {input_path}")
        
        # Get video info
        info = get_video_info(input_path)
        fps = info['fps']
        
        # Analyze audio
        plan = self.analyze_and_plan(input_path)
        print(f"[EffectsEngine] Effect plan: {len(plan.triggers)} triggers")
        
        # Process in batches
        processed_batches = []
        
        for batch, _ in decode_video_batched(input_path, batch_size=50, device=self.device):
            # Apply effects
            processed = self.effects_processor.process_with_plan(batch, plan, fps, batch_size=50)
            
            # Optional vignette
            if apply_vignette:
                processed = self.effects_processor.apply_vignette(processed.to(self.device), strength=0.3)
            
            processed_batches.append(processed.cpu())
            clear_gpu_cache()
        
        all_frames = torch.cat(processed_batches, dim=0)
        
        # Encode output
        success = encode_tensor_to_video(
            all_frames.to(self.device),
            output_path,
            fps=fps,
            width=info['width'],
            height=info['height']
        )
        
        if success:
            print(f"[EffectsEngine] Saved to: {output_path}")
        else:
            print("[EffectsEngine] Encoding failed!")
        
        clear_gpu_cache()
        return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python effects_engine.py <video_path> [output_path]")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else f"{Path(input_video).stem}_effects.mp4"
    
    engine = ContextAwareEffectsEngine()
    
    # Just analyze for now (full processing requires gpu_video_utils integration)
    plan = engine.analyze_and_plan(input_video)
    
    print(f"\n✅ Analysis complete!")
    print(f"Duration: {plan.duration:.1f}s")
    print(f"Average Energy: {plan.average_energy:.4f}")
    print(f"Peak moments: {len(plan.peak_timestamps)}")
    print(f"Effect triggers: {len(plan.triggers)}")
    
    # Save plan
    plan_path = f"{Path(input_video).stem}_effects_plan.json"
    engine.audio_analyzer.save_plan(plan, plan_path)
    
    print(f"\nEffect breakdown:")
    for effect_type in ['zoom', 'shake', 'emphasize', 'flash']:
        count = sum(1 for t in plan.triggers if t.effect_type == effect_type)
        if count > 0:
            print(f"  {effect_type}: {count} triggers")

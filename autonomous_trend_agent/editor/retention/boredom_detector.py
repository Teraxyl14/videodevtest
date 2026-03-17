"""
Boredom Detector: Retention Engineering Module
Analyzes video and audio for low-information density segments to trigger interventions.
"""

import cv2
import numpy as np
# import librosa <--- Moved to lazy import
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class BoredomDetector:
    """
    Detects segments with low visual or audio change (Boredom)
    and generates a Pacing Plan for the editor.
    """
    
    def __init__(
        self,
        ti_threshold: float = 5.0,     # Temporal Information threshold (pixel diff std dev)
        ae_threshold: float = 0.5,     # Audio Excitement threshold (onset strength)
        min_duration: float = 2.0,     # Minimum duration to trigger intervention
        zoom_intensity: float = 1.15   # 15% Punch-in
    ):
        self.ti_threshold = ti_threshold
        self.ae_threshold = ae_threshold
        self.min_duration = min_duration
        self.zoom_intensity = zoom_intensity
        
    def analyze(self, video_path: str, audio_path: Optional[str] = None) -> List[Dict]:
        """
        Analyze video for boredom points.
        
        Args:
            video_path: Path to video file
            audio_path: Optional separate audio path (uses video audio if None)
            
        Returns:
            List of interventions [{'type': 'zoom', 'start_time': X, 'end_time': Y, 'value': Z}]
        """
        video_path = str(video_path)
        print(f"[BoredomDetector] Analyzing {Path(video_path).name}...")
        
        # 1. calculate Temporal Information (TI) - Visual
        ti_scores, fps = self._calculate_ti(video_path)
        
        # 2. calculate Audio Excitement (AE) - Audio
        # Use video file for audio if not provided
        audio_src = audio_path if audio_path else video_path
        ae_scores, ae_sr = self._calculate_ae(audio_src, duration=len(ti_scores)/fps)
        
        if len(ti_scores) == 0:
            return []
            
        # Resample AE to match Video Frame Rate
        ae_resampled = np.interp(
            np.linspace(0, len(ae_scores), len(ti_scores)),
            np.arange(len(ae_scores)),
            ae_scores
        )
        
        # 3. Detect Low Density Zones
        # ARCHITECTURE NOTE: Short-Form Retention Theory
        # TikTok/Shorts viewers swipe away if stimulation drops for > 2 seconds.
        # This loop scans the synchronized TI/AE arrays. If BOTH visual variance
        # AND audio energy drop below their thresholds, it declares a 'Boredom Zone'
        # and schedules a +15% GPU Zoom Punch to artificially manipulate the frame 
        # and forcefully reset the viewer's attention span.
        
        interventions = []
        is_bored = False
        start_frame = 0
        
        # Sliding window analysis
        # We look for sustained periods where BOTH TI and AE are low
        
        # Normalize scores roughly to 0-1 for comparison? 
        # TI relies on absolute pixel diff, AE on normalized onsets.
        # Let's use the explicit thresholds provided.
        
        for i in range(len(ti_scores)):
            ti = ti_scores[i]
            ae = ae_resampled[i]
            
            # Condition: Visual is static AND Audio is calm
            # Note: During speech, AE might be high. We only zoom if speech is boring or Visual is VERY static.
            # Research: "If >3s with frame difference < 5%, insert Zoom".
            
            # Condition: Visual is static (ignoring subs) OR Audio is calm
            # If TI is low (static face), we zoom.
            # If AE is low (silence/boring speech), we zoom.
            
            # Simple logic: If visual is static, zoom.
            # If audio is low energy, zoom.
            # AND combined: ti < threshold OR ae < threshold
            
            # Revised for "Recursion": Subtitles might still leak into TI.
            # We trust AE more.
            condition = (ti < self.ti_threshold) or (ae < self.ae_threshold)
            
            if condition:
                if not is_bored:
                    is_bored = True
                    start_frame = i
            else:
                if is_bored:
                    is_bored = False
                    duration_sec = (i - start_frame) / fps
                    
                    if duration_sec > self.min_duration:
                        # Found a valid boredom segment
                        # Add intervention
                        interventions.append({
                            "type": "zoom",
                            "start_time": start_frame / fps,
                            "end_time": i / fps,
                            "value": self.zoom_intensity,
                            "reason": f"Low TI ({np.mean(ti_scores[start_frame:i]):.1f})"
                        })
                        
        # Check if ended in bored state
        if is_bored:
            duration_sec = (len(ti_scores) - start_frame) / fps
            if duration_sec > self.min_duration:
                 interventions.append({
                    "type": "zoom",
                    "start_time": start_frame / fps,
                    "end_time": len(ti_scores) / fps,
                    "value": self.zoom_intensity,
                    "reason": f"Low TI ({np.mean(ti_scores[start_frame:]):.1f})"
                })
        
        print(f"[BoredomDetector] Generated {len(interventions)} interventions")
        return interventions
    
    def _calculate_ti(self, video_path: str) -> Tuple[np.ndarray, float]:
        """Calculate Temporal Information (TI) using frame differencing"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[BoredomDetector] Error opening video {video_path}")
            return np.array([]), 30.0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Downscale for performance (320px width)
        scale = 320 / width
        new_w = 320
        new_h = int(height * scale)
        
        ti_scores = []
        prev_gray = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 2nd frame for speed (effective 15fps analysis)
            if frame_idx % 2 != 0:
                # Re-use previous score for this frame to maintain array length?
                # Or just duplicate.
                # Actually, let's process every frame but on small res.
                pass
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (new_w, new_h))
            
            # Crop bottom 40% to ignore subtitles/lower-thirds
            gray = gray[0:int(new_h * 0.6), :]
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                std_dev = np.std(diff)
                ti_scores.append(std_dev)
            else:
                ti_scores.append(0.0) # First frame
                
            prev_gray = gray
            frame_idx += 1
            
        cap.release()
        return np.array(ti_scores), fps
        
    def _calculate_ae(self, audio_path: str, duration: float) -> Tuple[np.ndarray, int]:
        """Calculate Audio Excitement (AE) using onset strength"""
        import numpy as np  # Ensure np is available locally to prevent shadowing/unbound errors
        try:
            import librosa
            # Load audio (mono)
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Calculate Onset Strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Flatten/Normalize
            # onset_env is per-frame (default hop_length=512)
            # 22050 / 512 = ~43 Hz
            
            return onset_env, sr
            
        except Exception as e:
            print(f"[BoredomDetector] Audio analysis failed: {e}")
            # Return dummy array of 1s (High Excitement) so we don't trigger boredom on error
            return np.ones(int(duration * 30)), 30

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python boredom_detector.py <video>")
        sys.exit(1)
        
    detector = BoredomDetector()
    plan = detector.analyze(sys.argv[1])
    
    import json
    print(json.dumps(plan, indent=2))

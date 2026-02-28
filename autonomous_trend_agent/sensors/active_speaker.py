"""
Active Speaker Detection
Identifies which person is speaking by correlating audio energy with face positions

Uses:
1. Voice Activity Detection (VAD) via librosa/webrtcvad
2. Face detection from YOLOv11 tracker
3. Audio-visual correlation to assign speaker labels

This enables the reframer to follow the active speaker in multi-person content.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
import subprocess


@dataclass
class SpeakerSegment:
    """A segment where a specific speaker is active"""
    speaker_id: int
    start_time: float
    end_time: float
    confidence: float
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class ActiveSpeakerResult:
    """Complete active speaker analysis result"""
    speakers: List[SpeakerSegment]
    num_speakers: int
    dominant_speaker_id: int
    speaking_time_per_speaker: Dict[int, float]
    video_path: str
    audio_path: Optional[str] = None


class VoiceActivityDetector:
    """
    Detect when speech is occurring using energy + spectral analysis.
    Falls back to simple energy thresholding if webrtcvad unavailable.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # 30ms frames
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Try to load webrtcvad for robust VAD
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
            self.use_webrtc = True
            print("[VAD] Using WebRTC VAD")
        except ImportError:
            self.vad = None
            self.use_webrtc = False
            print("[VAD] Using energy-based VAD (webrtcvad not available)")
    
    def detect(self, audio: np.ndarray, sample_rate: int = None) -> List[Dict]:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio samples (mono, float32)
            sample_rate: Sample rate (uses default if None)
            
        Returns:
            List of {"start_time": float, "end_time": float, "is_speech": bool}
        """
        sr = sample_rate or self.sample_rate
        
        if self.use_webrtc:
            return self._detect_webrtc(audio, sr)
        else:
            return self._detect_energy(audio, sr)
    
    def _detect_webrtc(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """WebRTC VAD detection"""
        import webrtcvad
        
        # Resample to 16kHz if needed (webrtcvad only supports 8k/16k/32k/48k)
        if sr not in [8000, 16000, 32000, 48000]:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        frame_size = int(sr * self.frame_duration_ms / 1000)
        segments = []
        
        current_speech = False
        speech_start = 0
        
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size].tobytes()
            is_speech = self.vad.is_speech(frame, sr)
            
            time = i / sr
            
            if is_speech and not current_speech:
                # Speech started
                speech_start = time
                current_speech = True
            elif not is_speech and current_speech:
                # Speech ended
                segments.append({
                    "start_time": speech_start,
                    "end_time": time,
                    "is_speech": True
                })
                current_speech = False
        
        # Handle trailing speech
        if current_speech:
            segments.append({
                "start_time": speech_start,
                "end_time": len(audio_int16) / sr,
                "is_speech": True
            })
        
        return segments
    
    def _detect_energy(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Energy-based VAD fallback"""
        import librosa
        
        # Compute RMS energy
        frame_length = int(sr * self.frame_duration_ms / 1000)
        hop_length = frame_length
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Compute threshold (adaptive)
        threshold = np.median(rms) * 2  # Speech typically 2x median energy
        
        segments = []
        current_speech = False
        speech_start = 0
        
        for i, energy in enumerate(rms):
            time = i * hop_length / sr
            is_speech = energy > threshold
            
            if is_speech and not current_speech:
                speech_start = time
                current_speech = True
            elif not is_speech and current_speech:
                if time - speech_start > 0.1:  # Minimum 100ms speech
                    segments.append({
                        "start_time": speech_start,
                        "end_time": time,
                        "is_speech": True
                    })
                current_speech = False
        
        if current_speech:
            end_time = len(rms) * hop_length / sr
            if end_time - speech_start > 0.1:
                segments.append({
                    "start_time": speech_start,
                    "end_time": end_time,
                    "is_speech": True
                })
        
        return segments


class ActiveSpeakerDetector:
    """
    Correlates voice activity with face positions to identify active speaker.
    
    Algorithm:
    1. Extract audio and detect voice activity periods
    2. Get face tracking data with bounding boxes
    3. For each speech segment, find face with highest motion/lip activity
    4. Assign speaker ID and build speaker timeline
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vad = VoiceActivityDetector()
        self._tracker = None
    
    @property
    def tracker(self):
        """Lazy load YOLO tracker"""
        if self._tracker is None:
            try:
                from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker
                self._tracker = YOLOv11Tracker(
                    device=self.device,
                    enable_pose=True  # Enable pose for Light-ASD (nose motion)
                )
            except ImportError:
                print("[ActiveSpeaker] YOLOv11Tracker not available, using face detection only")
        return self._tracker
    
    def analyze(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        face_data: Optional[Dict] = None
    ) -> ActiveSpeakerResult:
        """
        Analyze video to identify active speaker at each moment.
        
        Args:
            video_path: Path to video file
            audio_path: Optional separate audio file (extracts from video if None)
            face_data: Optional pre-computed face tracking data
            
        Returns:
            ActiveSpeakerResult with speaker segments
        """
        import librosa
        
        print(f"[ActiveSpeaker] Analyzing: {video_path}")
        
        # Step 1: Extract audio if needed
        if audio_path is None:
            audio_path = self._extract_audio(video_path)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"[ActiveSpeaker] Audio loaded: {len(audio)/sr:.1f}s @ {sr}Hz")
        
        # Step 2: Detect voice activity
        speech_segments = self.vad.detect(audio, sr)
        print(f"[ActiveSpeaker] Found {len(speech_segments)} speech segments")
        
        if not speech_segments:
            return self._empty_result(video_path, audio_path)
        
        # Step 3: Get face tracking data
        if face_data is None:
            face_data = self._track_faces(video_path)
        
        # Step 4: Correlate speech with faces
        speaker_segments = self._correlate_speech_faces(
            speech_segments,
            face_data,
            audio,
            sr
        )
        
        # Step 5: Build result
        return self._build_result(speaker_segments, video_path, audio_path)
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using FFmpeg"""
        output_path = Path(video_path).with_suffix('.wav')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz for VAD
            '-ac', '1',  # Mono
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return str(output_path)
    
    def _track_faces(self, video_path: str) -> Dict:
        """Get face tracking data"""
        if self.tracker is not None:
            result = self.tracker.track_video(
                video_path,
                target_class="face",
                smooth=True
            )
            return result
        
        # Fallback: simple face detection with OpenCV
        return self._opencv_face_detection(video_path)
    
    def _opencv_face_detection(self, video_path: str) -> Dict:
        """Fallback face detection using OpenCV Haar Cascades"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces_by_frame = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces every 5 frames for speed
            if frame_idx % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                faces_by_frame.append({
                    "frame_idx": frame_idx,
                    "time": frame_idx / fps,
                    "faces": [
                        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                        for x, y, w, h in faces
                    ]
                })
            
            frame_idx += 1
        
        cap.release()
        
        return {
            "fps": fps,
            "faces_by_frame": faces_by_frame,
            "tracked_objects": []  # Not using tracking, just detection
        }
    
    def _correlate_speech_faces(
        self,
        speech_segments: List[Dict],
        face_data: Dict,
        audio: np.ndarray,
        sr: int
    ) -> List[SpeakerSegment]:
        """
        Correlate speech segments with detected faces.
        Uses motion analysis to identify the speaking face.
        """
        import librosa
        
        speakers = []
        fps = face_data.get("fps", 30)
        
        # Get tracked objects or faces_by_frame
        tracked = face_data.get("tracked_objects", [])
        faces_by_frame = face_data.get("faces_by_frame", [])
        
        for seg in speech_segments:
            start = seg["start_time"]
            end = seg["end_time"]
            
            # Find faces in this time window
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            
            # Compute audio energy for this segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Get energy profile
            rms = librosa.feature.rms(y=segment_audio, frame_length=512, hop_length=256)[0]
            
            # Find active face (one with most motion or closest to center)
            best_face = None
            best_score = -1
            
            if tracked:
                # Use YOLO tracking data
                for obj in tracked:
                    traj = obj.get("trajectory", [])
                    
                    # Filter trajectory to this segment
                    seg_traj = [
                        p for p in traj
                        if start_frame <= p.get("frame_idx", 0) <= end_frame
                    ]
                    
                    if len(seg_traj) < 2:
                        continue
                    
                    # Compute face motion (lip movement proxy)
                    motion = self._compute_motion(seg_traj)
                    
                    # Score = motion * audio correlation
                    score = motion * np.mean(rms)
                    
                    if score > best_score:
                        best_score = score
                        best_face = obj
            
            elif faces_by_frame:
                # Use frame-by-frame detection
                # Pick face closest to center (simple heuristic)
                seg_frames = [
                    f for f in faces_by_frame
                    if start <= f.get("time", 0) <= end
                ]
                
                if seg_frames and seg_frames[0].get("faces"):
                    # Use first detected face (simplest case)
                    best_face = seg_frames[0]["faces"][0]
            
            # Create speaker segment
            speaker_id = 0  # Default to single speaker
            bbox = None
            
            if best_face:
                # Try to get unique ID from tracking
                speaker_id = best_face.get("track_id", hash(str(best_face.get("x", 0))) % 10)
                
                if "trajectory" in best_face and best_face["trajectory"]:
                    mid_traj = best_face["trajectory"][len(best_face["trajectory"]) // 2]
                    bbox = mid_traj.get("bbox")
                elif "x" in best_face:
                    bbox = (best_face["x"], best_face["y"], 
                            best_face["w"], best_face["h"])
            
            speakers.append(SpeakerSegment(
                speaker_id=speaker_id,
                start_time=start,
                end_time=end,
                confidence=min(best_score / 10, 1.0) if best_score > 0 else 0.5,
                face_bbox=bbox
            ))
        
        return speakers
    
    def _compute_motion(self, trajectory: List[Dict]) -> float:
        """
        Compute motion amount from trajectory (proxy for lip/head movement).
        Uses Pose Keypoints (Light-ASD) if available, otherwise BBox center.
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_motion = 0
        valid_points = 0
        
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            
            # Strategy 1: Use Pose (Nose/Head) - Most Accurate
            prev_pt = None
            curr_pt = None
            
            if prev.get("pose") and curr.get("pose"):
                # Try nose first
                if "nose" in prev["pose"] and "nose" in curr["pose"]:
                    prev_pt = (prev["pose"]["nose"]["x"], prev["pose"]["nose"]["y"])
                    curr_pt = (curr["pose"]["nose"]["x"], curr["pose"]["nose"]["y"])
                # Fallback to head center
                elif "head_center" in prev["pose"] and "head_center" in curr["pose"]:
                    prev_pt = (prev["pose"]["head_center"]["x"], prev["pose"]["head_center"]["y"])
                    curr_pt = (curr["pose"]["head_center"]["x"], curr["pose"]["head_center"]["y"])
            
            # Strategy 2: Use BBox Center - Fallback
            if prev_pt is None:
                prev_cx = prev.get("x", 0) + prev.get("w", 0) / 2
                prev_cy = prev.get("y", 0) + prev.get("h", 0) / 2
                curr_cx = curr.get("x", 0) + curr.get("w", 0) / 2
                curr_cy = curr.get("y", 0) + curr.get("h", 0) / 2
                
                prev_pt = (prev_cx, prev_cy)
                curr_pt = (curr_cx, curr_cy)
            
            # Calculate Euclidean distance
            motion = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)
            total_motion += motion
            valid_points += 1
        
        return total_motion / valid_points if valid_points > 0 else 0.0
    
    def _empty_result(self, video_path: str, audio_path: str) -> ActiveSpeakerResult:
        """Return empty result when no speech detected"""
        return ActiveSpeakerResult(
            speakers=[],
            num_speakers=0,
            dominant_speaker_id=-1,
            speaking_time_per_speaker={},
            video_path=video_path,
            audio_path=audio_path
        )
    
    def _build_result(
        self,
        segments: List[SpeakerSegment],
        video_path: str,
        audio_path: str
    ) -> ActiveSpeakerResult:
        """Build final result with statistics"""
        if not segments:
            return self._empty_result(video_path, audio_path)
        
        # Count unique speakers
        speaker_ids = set(s.speaker_id for s in segments)
        
        # Compute speaking time per speaker
        time_per_speaker = {}
        for seg in segments:
            if seg.speaker_id not in time_per_speaker:
                time_per_speaker[seg.speaker_id] = 0
            time_per_speaker[seg.speaker_id] += seg.duration()
        
        # Find dominant speaker
        dominant = max(time_per_speaker.keys(), key=lambda k: time_per_speaker[k])
        
        print(f"[ActiveSpeaker] Detected {len(speaker_ids)} speakers, dominant: {dominant}")
        
        return ActiveSpeakerResult(
            speakers=segments,
            num_speakers=len(speaker_ids),
            dominant_speaker_id=dominant,
            speaking_time_per_speaker=time_per_speaker,
            video_path=video_path,
            audio_path=audio_path
        )
    
    def get_speaker_at_time(
        self,
        result: ActiveSpeakerResult,
        time: float
    ) -> Optional[SpeakerSegment]:
        """Get the active speaker at a specific time"""
        for seg in result.speakers:
            if seg.start_time <= time <= seg.end_time:
                return seg
        return None
    
    def save_result(self, result: ActiveSpeakerResult, output_path: str):
        """Save result to JSON"""
        data = {
            "num_speakers": result.num_speakers,
            "dominant_speaker_id": result.dominant_speaker_id,
            "speaking_time_per_speaker": result.speaking_time_per_speaker,
            "video_path": result.video_path,
            "segments": [
                {
                    "speaker_id": s.speaker_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "confidence": s.confidence,
                    "face_bbox": s.face_bbox
                }
                for s in result.speakers
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[ActiveSpeaker] Saved to: {output_path}")


def detect_active_speaker(
    video_path: str,
    audio_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> ActiveSpeakerResult:
    """
    Convenience function to detect active speaker.
    
    Args:
        video_path: Path to video
        audio_path: Optional separate audio file
        output_path: Optional path to save JSON result
        
    Returns:
        ActiveSpeakerResult
    """
    detector = ActiveSpeakerDetector()
    result = detector.analyze(video_path, audio_path)
    
    if output_path:
        detector.save_result(result, output_path)
    
    return result


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("ACTIVE SPEAKER DETECTION")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python active_speaker.py <video_path> [output.json]")
        print("\nDetects which person is speaking in multi-speaker video.")
        sys.exit(1)
    
    video = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = detect_active_speaker(video, output_path=output)
    
    print(f"\n✅ Detected {result.num_speakers} speaker(s)")
    print(f"   Dominant speaker ID: {result.dominant_speaker_id}")
    print(f"   Speech segments: {len(result.speakers)}")
    
    for sid, duration in result.speaking_time_per_speaker.items():
        print(f"   Speaker {sid}: {duration:.1f}s")

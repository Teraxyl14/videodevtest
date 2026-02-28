import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

HAS_FILTERPY = False
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    HAS_FILTERPY = True
except ImportError:
    KalmanFilter = None
    Q_discrete_white_noise = None

class ViralEngine:
    def __init__(self):
        self.reframer = KalmanReframer()
        self.boredom_detector = BoredomDetector()
        self.sync_engine = PerceptualSync()
        self.logger = logging.getLogger("ViralEngine")

@dataclass
class FrameCrop:
    x: int
    y: int
    w: int
    h: int
    score: float

class KalmanReframer:
    """
    Implements smooth cinematic reframing using Kalman Filtering.
    Targets the 'Rule of Thirds' (eye-line at 33% height).
    Falls back to exponential smoothing if filterpy is not installed.
    """
    def __init__(self, fps=30):
        self._use_kalman = HAS_FILTERPY
        self._smooth_x = None
        self._smooth_y = None
        self._alpha = 0.15  # Exponential smoothing factor (fallback)

        if self._use_kalman:
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = np.array([[1, 1, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]])
            self.kf.H = np.array([[1, 0, 0, 0],
                                  [0, 0, 1, 0]])
            self.kf.R *= 10
            self.kf.P *= 1000
            self.kf.Q = Q_discrete_white_noise(dim=4, dt=1/fps, var=0.1)
        else:
            self.kf = None

    def update(self, detection_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Updates the filter with a new face detection and returns smoothed coordinates.
        Format: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = detection_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if self._use_kalman and self.kf is not None:
            self.kf.predict()
            self.kf.update([center_x, center_y])
            return int(self.kf.x[0]), int(self.kf.x[2])
        else:
            # Exponential smoothing fallback
            if self._smooth_x is None:
                self._smooth_x = center_x
                self._smooth_y = center_y
            else:
                self._smooth_x = self._alpha * center_x + (1 - self._alpha) * self._smooth_x
                self._smooth_y = self._alpha * center_y + (1 - self._alpha) * self._smooth_y
            return int(self._smooth_x), int(self._smooth_y)

class BoredomDetector:
    """
    Analyzes visual and audio density to detect 'boring' segments.
    """
    def __init__(self, threshold_ti=5.0, threshold_ae=0.1):
        self.threshold_ti = threshold_ti # Temporal Information (Motion)
        self.threshold_ae = threshold_ae # Audio Excitement (RMS)

    def analyze_segment(self, visual_ti: List[float], audio_rms: List[float]) -> List[Tuple[int, str]]:
        """
        Returns a list of timestamps and interventions (e.g., "ZOOM", "CUT")
        """
        interventions = []
        for i, (ti, ae) in enumerate(zip(visual_ti, audio_rms)):
            if ti < self.threshold_ti and ae < self.threshold_ae:
                interventions.append((i, "ZOOM_IN")) # Add motion if none exists
        return interventions

class PerceptualSync:
    """
    Handles the -50ms offset for subtitles to match brain processing time.
    """
    def sync_subtitles(self, subtitles: List[dict], offset_ms: int = -50) -> List[dict]:
        """
        Shifts subtitle start/end times by offset_ms.
        """
        synced = []
        for sub in subtitles:
            start = max(0, sub['start'] + offset_ms)
            end = max(0, sub['end'] + offset_ms)
            synced.append({
                'text': sub['text'],
                'start': start,
                'end': end
            })
        return synced

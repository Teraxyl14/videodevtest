"""
Sensors module - perception and detection utilities
"""

from .active_speaker import (
    ActiveSpeakerDetector,
    ActiveSpeakerResult,
    SpeakerSegment,
    VoiceActivityDetector,
    detect_active_speaker
)

__all__ = [
    'ActiveSpeakerDetector',
    'ActiveSpeakerResult',
    'SpeakerSegment',
    'VoiceActivityDetector',
    'detect_active_speaker'
]

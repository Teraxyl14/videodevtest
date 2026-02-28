"""
Captions module - animated word-by-word subtitles
"""

from .animated_captions import (
    AnimatedCaptionEngine,
    CaptionStyle,
    CAPTION_STYLES,
    add_animated_captions
)

# Re-export from caption_engine if it exists
try:
    from .caption_engine import CaptionEngine
except ImportError:
    pass

__all__ = [
    'AnimatedCaptionEngine',
    'CaptionStyle', 
    'CAPTION_STYLES',
    'add_animated_captions',
]

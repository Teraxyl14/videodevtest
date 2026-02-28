"""
Context-Aware Effects Module
Audio-driven visual effects using librosa + Kornia
"""

from .effects_engine import (
    AudioAnalyzer,
    GPUEffectsProcessor,
    ContextAwareEffectsEngine,
    EffectTrigger,
    EffectPlan
)

__all__ = [
    'AudioAnalyzer',
    'GPUEffectsProcessor', 
    'ContextAwareEffectsEngine',
    'EffectTrigger',
    'EffectPlan'
]

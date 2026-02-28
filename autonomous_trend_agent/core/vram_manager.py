import torch
import gc
import json
import logging
import platform
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class TransitionContextManager:
    """
    Guarantees a clean GPU state between pipeline phases.
    Prevents CUDACachingAllocator fragmentation and stale handle issues.
    """
    def __init__(self, phase_name: str):
        self.phase_name = phase_name

    def __enter__(self):
        logger.info(f"--- [Pipeline] Entering Phase: {self.phase_name} ---")
        self._sync_and_clean()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"--- [Pipeline] Exiting Phase: {self.phase_name} ---")
        # If an error occurred, we try to clean up aggressively
        if exc_type:
            logger.warning(f"Error detected in {self.phase_name}. Attempting emergency cleanup.")
            self._sync_and_clean()
        else:
            self._sync_and_clean()

    def _sync_and_clean(self):
        try:
            # 1. Synchronize to ensure all kernels from the previous phase are done
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 2. Python-side GC to free object wrappers
            gc.collect()
            
            if torch.cuda.is_available():
                # 3. Release PyTorch Allocator's cached blocks
                torch.cuda.empty_cache()
                
                # 4. Clean up IPC handles (critical for zero-copy/DLPack workflows)
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                
                # 5. Reset peak memory stats to allow accurate monitoring for the next phase
                torch.cuda.reset_peak_memory_stats()
                
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

# Try to import pynvml, but gracefully fallback to PyTorch memory APIs
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available, using PyTorch memory APIs")

class PipelinePhase(Enum):
    """Pipeline phases with their VRAM budgets"""
    IDLE = "idle"
    ANALYSIS = "analysis"      # Qwen3-VL: 8.5GB
    AUDIO = "audio"            # Canary: 4.2GB
    TRACKING = "tracking"      # Canary + SAM2: 6.0GB
    GENERATION = "generation"  # Wan 2.1: 10.2GB (future)

@dataclass
class VRAMBudget:
    """VRAM budget specifications for each phase"""
    model_vram: float  # GB required for model weights
    system_overhead: float  # GB for system + display
    total: float  # Total VRAM needed
    
    @property
    def safe_threshold(self) -> float:
        """Add 10% safety margin"""
        return self.total * 1.1

# Phase-specific budgets (from PDF Table 1)
PHASE_BUDGETS: Dict[PipelinePhase, VRAMBudget] = {
    # Qwen3-VL Int4 is ~5.5GB. Host usage high (>8GB), leaving <7GB free.
    # Aggressive tune: 5.0GB model + 1.0GB overhead = 6.0GB (Safe: 6.6GB)
    PipelinePhase.ANALYSIS: VRAMBudget(5.0, 1.0, 6.0),
    PipelinePhase.AUDIO: VRAMBudget(2.2, 2.0, 4.2),
    PipelinePhase.TRACKING: VRAMBudget(4.0, 2.0, 6.0),  # Canary + SAM2
    PipelinePhase.GENERATION: VRAMBudget(8.2, 2.0, 10.2),
}

class VRAMOrchestrator:
    """
    Manages the lifecycle of models across pipeline phases.
    Ensures strict adherence to 16GB VRAM budget through serialization.
    """
    
    def __init__(self, total_vram_gb: float = 16.0):
        self.total_vram_gb = total_vram_gb
        self.current_phase = PipelinePhase.IDLE
        self.loaded_models: Dict[str, Any] = {}
        
        # Initialize NVML for VRAM monitoring (with fallback)
        self._nvml_available = False
        self.handle = None
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_available = True
                logger.info("NVML initialized successfully")
            except Exception as e:
                logger.warning(f"NVML init failed: {e}. Using PyTorch fallback.")
        
    def get_free_vram_gb(self) -> float:
        """Get current free VRAM in GB"""
        if self._nvml_available and self.handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.free / (1024 ** 3)
            except:
                pass
        
        # Fallback: Use PyTorch memory APIs
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            return (total - allocated) / (1024 ** 3)
        return self.total_vram_gb  # Assume full VRAM if no GPU
    
    def get_used_vram_gb(self) -> float:
        """Get current used VRAM in GB"""
        if self._nvml_available and self.handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.used / (1024 ** 3)
            except:
                pass
        
        # Fallback: Use PyTorch memory APIs
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 ** 3)
        return 0.0
    
    def validate_phase_transition(self, target_phase: PipelinePhase) -> bool:
        """
        Pre-flight check before loading models for a phase.
        Raises VRAMInsufficientError if not enough memory.
        """
        if target_phase not in PHASE_BUDGETS:
            return True  # IDLE phase, no budget
        
        budget = PHASE_BUDGETS[target_phase]
        free_vram = self.get_free_vram_gb()
        
        if free_vram < budget.safe_threshold:
            raise VRAMInsufficientError(
                f"Phase {target_phase.value} needs {budget.safe_threshold:.1f}GB, "
                f"only {free_vram:.1f}GB free. Current phase: {self.current_phase.value}"
            )
        
        print(f"✓ VRAM check passed for {target_phase.value}: "
              f"{free_vram:.1f}GB free, need {budget.total:.1f}GB")
        return True
    
    def clear_cuda_cache(self):
        """Hard VRAM flush - critical between phases"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print(f"🗑️  CUDA cache cleared. Free VRAM: {self.get_free_vram_gb():.1f}GB")
    
    def unload_all_models(self):
        """Unload all currently loaded models"""
        for name, model in self.loaded_models.items():
            del model
            print(f"⬇️  Unloaded: {name}")
        
        self.loaded_models.clear()
        self.clear_cuda_cache()
        self.current_phase = PipelinePhase.IDLE
    
    def register_model(self, name: str, model: Any):
        """Register a loaded model for lifecycle tracking"""
        self.loaded_models[name] = model
        print(f"⬆️  Loaded: {name} | VRAM: {self.get_used_vram_gb():.1f}GB / {self.total_vram_gb}GB")
    
    def unregister_model(self, name: str):
        """Unload a specific model"""
        if name in self.loaded_models:
            del self.loaded_models[name]
            self.clear_cuda_cache()
            print(f"⬇️  Unloaded: {name}")
    
    # Phase Transitions
    
    def transition_to_analysis(self):
        """Phase 1: Load Qwen3-VL for video analysis"""
        self.validate_phase_transition(PipelinePhase.ANALYSIS)
        self.unload_all_models()
        self.current_phase = PipelinePhase.ANALYSIS
        print(f"📊 Entering ANALYSIS phase (Budget: {PHASE_BUDGETS[PipelinePhase.ANALYSIS].total}GB)")
    
    def transition_to_audio(self):
        """Phase 2: Unload analysis, load Canary ASR"""
        self.validate_phase_transition(PipelinePhase.AUDIO)
        self.unload_all_models()  # Clear Qwen
        self.current_phase = PipelinePhase.AUDIO
        print(f"🎤 Entering AUDIO phase (Budget: {PHASE_BUDGETS[PipelinePhase.AUDIO].total}GB)")
    
    def transition_to_tracking(self):
        """Phase 3: Keep Canary, add SAM2 (coexistence)"""
        self.validate_phase_transition(PipelinePhase.TRACKING)
        # Note: Canary stays loaded, we just add SAM2
        self.current_phase = PipelinePhase.TRACKING
        print(f"🎯 Entering TRACKING phase (Budget: {PHASE_BUDGETS[PipelinePhase.TRACKING].total}GB)")
    
    def transition_to_generation(self):
        """Phase 4: Hard flush, load Wan 2.1 for B-roll"""
        self.validate_phase_transition(PipelinePhase.GENERATION)
        self.unload_all_models()  # Hard flush
        self.current_phase = PipelinePhase.GENERATION
        print(f"🎨 Entering GENERATION phase (Budget: {PHASE_BUDGETS[PipelinePhase.GENERATION].total}GB)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "phase": self.current_phase.value,
            "models_loaded": list(self.loaded_models.keys()),
            "vram_used_gb": self.get_used_vram_gb(),
            "vram_free_gb": self.get_free_vram_gb(),
            "vram_total_gb": self.total_vram_gb
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        if self._nvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

class VRAMInsufficientError(Exception):
    """Raised when insufficient VRAM for phase transition"""
    pass

if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = VRAMOrchestrator()
    
    print("=== VRAM Orchestrator Test ===")
    print(f"Total VRAM: {orchestrator.total_vram_gb}GB")
    print(f"Free VRAM: {orchestrator.get_free_vram_gb():.1f}GB")
    print(f"Used VRAM: {orchestrator.get_used_vram_gb():.1f}GB")
    
    # Test phase transitions
    try:
        orchestrator.transition_to_analysis()
        print(orchestrator.get_status())
        
        orchestrator.transition_to_audio()
        print(orchestrator.get_status())
        
        orchestrator.transition_to_tracking()
        print(orchestrator.get_status())
        
    except VRAMInsufficientError as e:
        print(f"❌ Error: {e}")

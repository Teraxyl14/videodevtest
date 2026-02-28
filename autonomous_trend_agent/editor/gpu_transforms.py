"""
GPU Transforms - Kornia-based Image Processing
===============================================
GPU-accelerated image transforms using Kornia.
Eliminates CPU bottleneck for preprocessing and effects.

Features:
- GPU-native resizing, cropping, color conversion
- Context-aware effects (zoom, shake)
- All operations stay in VRAM (zero-copy compatible)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import kornia, fallback to torch operations
try:
    import kornia
    import kornia.geometry.transform as KT
    import kornia.filters as KF
    import kornia.enhance as KE
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    logger.warning("Kornia not available, using torch fallback transforms")


class GPUTransforms:
    """
    GPU-accelerated image transforms for video processing.
    
    All operations work on GPU tensors and keep data in VRAM.
    Compatible with zero-copy pipeline (PyNvVideoCodec + DLPack).
    
    Usage:
        transforms = GPUTransforms(device="cuda")
        resized = transforms.resize(frame, (1920, 1080))
        cropped = transforms.crop(frame, x, y, w, h)
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_kornia = KORNIA_AVAILABLE
    
    def resize(
        self,
        tensor: torch.Tensor,
        size: Tuple[int, int],
        mode: str = "bilinear"
    ) -> torch.Tensor:
        """
        Resize tensor to target size.
        
        Args:
            tensor: (H, W, C) or (B, C, H, W) tensor
            size: (height, width) target size
            mode: "bilinear", "nearest", or "bicubic"
        
        Returns:
            Resized tensor
        """
        # Convert HWC to BCHW if needed
        squeeze = False
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            squeeze = True
        
        if self.use_kornia:
            result = KT.resize(tensor.float(), size, interpolation=mode)
        else:
            result = F.interpolate(tensor.float(), size=size, mode=mode, align_corners=False)
        
        if squeeze:
            result = result.squeeze(0).permute(1, 2, 0)
        
        return result.to(tensor.dtype)
    
    def crop(
        self,
        tensor: torch.Tensor,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> torch.Tensor:
        """
        Crop a region from the tensor.
        
        Args:
            tensor: (H, W, C) tensor
            x, y: Top-left corner
            width, height: Crop size
        
        Returns:
            Cropped tensor
        """
        # Clamp to valid bounds
        h, w = tensor.shape[:2]
        x = max(0, min(x, w - width))
        y = max(0, min(y, h - height))
        
        return tensor[y:y+height, x:x+width]
    
    def center_crop(
        self,
        tensor: torch.Tensor,
        size: Tuple[int, int]
    ) -> torch.Tensor:
        """Center crop to target size."""
        h, w = tensor.shape[:2]
        th, tw = size
        
        x = (w - tw) // 2
        y = (h - th) // 2
        
        return self.crop(tensor, x, y, tw, th)
    
    def to_vertical(
        self,
        tensor: torch.Tensor,
        center_x: Optional[float] = None
    ) -> torch.Tensor:
        """
        Convert horizontal video frame to vertical (9:16).
        
        Args:
            tensor: (H, W, C) horizontal frame (16:9)
            center_x: Relative x position to center crop (0-1)
        
        Returns:
            Vertical frame (9:16)
        """
        h, w = tensor.shape[:2]
        
        # Calculate vertical frame size
        target_w = int(h * 9 / 16)
        
        if center_x is None:
            center_x = 0.5
        
        # Calculate crop position
        x = int((w - target_w) * center_x)
        
        return self.crop(tensor, x, 0, target_w, h)
    
    def apply_zoom(
        self,
        tensor: torch.Tensor,
        zoom_factor: float,
        center: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Apply zoom effect centered on a point.
        
        Args:
            tensor: Input frame
            zoom_factor: 1.0 = no zoom, 1.2 = 20% zoom in
            center: Relative center point (0-1, 0-1), default center
        
        Returns:
            Zoomed frame (same size as input)
        """
        h, w = tensor.shape[:2]
        
        if center is None:
            center = (0.5, 0.5)
        
        # Calculate crop size for zoom
        crop_h = int(h / zoom_factor)
        crop_w = int(w / zoom_factor)
        
        # Calculate top-left corner
        x = int((w - crop_w) * center[0])
        y = int((h - crop_h) * center[1])
        
        # Crop and resize back
        cropped = self.crop(tensor, x, y, crop_w, crop_h)
        return self.resize(cropped, (h, w))
    
    def apply_shake(
        self,
        tensor: torch.Tensor,
        intensity: float,
        offset: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Apply camera shake effect.
        
        Args:
            tensor: Input frame
            intensity: Shake intensity (0-1)
            offset: Random offset for this frame (-1 to 1)
        
        Returns:
            Shaken frame
        """
        h, w = tensor.shape[:2]
        
        # Calculate pixel offset
        max_offset = int(min(w, h) * intensity * 0.02)
        dx = int(offset[0] * max_offset)
        dy = int(offset[1] * max_offset)
        
        # Pad and crop to create shake
        padded = F.pad(
            tensor.permute(2, 0, 1).unsqueeze(0).float(),
            (max_offset, max_offset, max_offset, max_offset),
            mode='reflect'
        )
        
        x = max_offset + dx
        y = max_offset + dy
        
        result = padded[:, :, y:y+h, x:x+w]
        return result.squeeze(0).permute(1, 2, 0).to(tensor.dtype)
    
    def normalize(
        self,
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> torch.Tensor:
        """Normalize tensor for model input."""
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        
        if self.use_kornia:
            return KE.normalize(
                tensor.permute(2, 0, 1).unsqueeze(0),
                torch.tensor(mean),
                torch.tensor(std)
            ).squeeze(0).permute(1, 2, 0)
        
        mean_t = torch.tensor(mean, device=tensor.device)
        std_t = torch.tensor(std, device=tensor.device)
        return (tensor - mean_t) / std_t
    
    def denormalize(
        self,
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> torch.Tensor:
        """Denormalize tensor back to 0-255 range."""
        mean_t = torch.tensor(mean, device=tensor.device)
        std_t = torch.tensor(std, device=tensor.device)
        
        tensor = tensor * std_t + mean_t
        tensor = (tensor * 255).clamp(0, 255)
        
        return tensor.to(torch.uint8)
    
    def gaussian_blur(
        self,
        tensor: torch.Tensor,
        kernel_size: int = 5,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """Apply Gaussian blur."""
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        if self.use_kornia:
            result = KF.gaussian_blur2d(tensor.float(), (kernel_size, kernel_size), (sigma, sigma))
        else:
            # Simple box blur fallback
            result = F.avg_pool2d(
                tensor.float(),
                kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
        
        if squeeze:
            result = result.squeeze(0).permute(1, 2, 0)
        
        return result.to(tensor.dtype)
    
    def adjust_brightness(
        self,
        tensor: torch.Tensor,
        factor: float
    ) -> torch.Tensor:
        """Adjust brightness (factor > 1 = brighter)."""
        if self.use_kornia:
            if tensor.dim() == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                result = KE.adjust_brightness(tensor.float() / 255.0, factor)
                return (result.squeeze(0).permute(1, 2, 0) * 255).to(tensor.dtype)
        
        return (tensor.float() * factor).clamp(0, 255).to(tensor.dtype)
    
    def adjust_contrast(
        self,
        tensor: torch.Tensor,
        factor: float
    ) -> torch.Tensor:
        """Adjust contrast (factor > 1 = more contrast)."""
        if self.use_kornia:
            if tensor.dim() == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                result = KE.adjust_contrast(tensor.float() / 255.0, factor)
                return (result.squeeze(0).permute(1, 2, 0) * 255).to(tensor.dtype)
        
        mean = tensor.float().mean()
        return ((tensor.float() - mean) * factor + mean).clamp(0, 255).to(tensor.dtype)


class ContextAwareEffects:
    """
    Context-aware visual effects triggered by audio analysis.
    
    Uses RMS energy and prosody data to apply zoom/shake at key moments.
    """
    
    def __init__(self):
        self.transforms = GPUTransforms()
    
    def apply_emphasis_zoom(
        self,
        frame: torch.Tensor,
        rms_energy: float,
        threshold: float = 0.5,
        max_zoom: float = 1.15
    ) -> torch.Tensor:
        """
        Apply zoom on emphasized words (high RMS energy).
        
        Args:
            frame: Input frame
            rms_energy: Normalized RMS energy (0-1)
            threshold: Minimum energy to trigger zoom
            max_zoom: Maximum zoom factor
        
        Returns:
            Zoomed frame if energy exceeds threshold
        """
        if rms_energy < threshold:
            return frame
        
        # Scale zoom based on energy
        zoom_factor = 1.0 + (rms_energy - threshold) * (max_zoom - 1.0) / (1.0 - threshold)
        return self.transforms.apply_zoom(frame, zoom_factor)
    
    def apply_impact_shake(
        self,
        frame: torch.Tensor,
        frame_idx: int,
        impact_frames: List[int],
        intensity: float = 0.3,
        duration: int = 5
    ) -> torch.Tensor:
        """
        Apply screen shake at impact moments.
        
        Args:
            frame: Input frame
            frame_idx: Current frame index
            impact_frames: List of frame indices with high impact
            intensity: Shake intensity
            duration: Number of frames to shake
        
        Returns:
            Shaken frame if near impact point
        """
        for impact in impact_frames:
            if impact <= frame_idx < impact + duration:
                # Decay shake over duration
                t = (frame_idx - impact) / duration
                current_intensity = intensity * (1 - t)
                
                # Generate pseudo-random offset
                import math
                offset = (
                    math.sin(frame_idx * 7.3) * 2 - 1,
                    math.cos(frame_idx * 11.7) * 2 - 1
                )
                
                return self.transforms.apply_shake(frame, current_intensity, offset)
        
        return frame

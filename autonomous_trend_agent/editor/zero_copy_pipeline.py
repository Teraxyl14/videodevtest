"""
Zero-Copy GPU Video Pipeline with Dynamic Tracking

Full production pipeline that:
1. Decodes video using NVDEC → GPU memory
2. Applies per-frame dynamic cropping using KalmanCamera  
3. Transforms using Kornia (GPU-native)
4. Encodes using NVENC
5. Muxes audio from original

This is the production-quality solution per research findings.
"""

import subprocess
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
import tempfile
import shutil

from .gpu_video_utils import get_video_info, clear_gpu_cache


# -----------------------------------------------------------------------------
# OneEuroFilter Implementation (Self-Contained)
# -----------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        t_e = t - self.t_prev
        
        # Avoid division by zero on very small time steps (or duplicate timestamps)
        if t_e <= 0.0:
            return x

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.dx_prev + a_d * (dx - self.dx_prev)
        
        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.x_prev + a * (x - self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat


class OneEuroCamera:
    """Camera smoothing using OneEuroFilter"""
    def __init__(self, src_w, src_h, crop_w, crop_h, min_cutoff=0.005, beta=0.002): # Tuned for ultra-smooth motion (v2)
        self.filters = {}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.last_t = -1.0
        self.src_w = src_w
        self.src_h = src_h
        # We don't necessarily use crop_w/crop_h here but store for ref if needed
        self.base_crop_w = crop_w
        self.base_crop_h = crop_h
    
    def update(self, t: float, cx: float, cy: float, zoom: float) -> Tuple[float, float, float]:
        if "cx" not in self.filters:
            self.filters["cx"] = OneEuroFilter(t, cx, min_cutoff=self.min_cutoff, beta=self.beta)
            self.filters["cy"] = OneEuroFilter(t, cy, min_cutoff=self.min_cutoff, beta=self.beta)
            self.filters["zoom"] = OneEuroFilter(t, zoom, min_cutoff=self.min_cutoff, beta=self.beta)
            return cx, cy, zoom
        
        # Prevent update with same timestamp
        if t <= self.last_t:
            t = self.last_t + 0.001
            
        cx_smooth = self.filters["cx"].filter(t, cx)
        cy_smooth = self.filters["cy"].filter(t, cy)
        zoom_smooth = self.filters["zoom"].filter(t, zoom)
        
        self.last_t = t
        return cx_smooth, cy_smooth, zoom_smooth

    def reset(self, cx: int, cy: int):
        """Reset filters to a new starting position"""
        self.filters = {}
        self.last_t = -1.0
        # The next update() call will automatically re-initialize filters at (cx, cy)
    
    def get_crop(self, target_cx: int, target_cy: int, crop_w: int, crop_h: int) -> Tuple[int, int, int, int]:
        """
        Get smoothed crop coordinates (top-left x, y) and smoothed dimensions (w, h).
        
        Args:
            target_cx: Target center X
            target_cy: Target center Y
            crop_w: Width of crop window (from zoom map)
            crop_h: Height of crop window (from zoom map)
            
        Returns:
            (x, y, w, h) ensuring bounds and smoothing
        """
        if self.last_t < 0:
            self.last_t = 0.0
            
        # Pseudo-timestamp increment
        t = self.last_t + (1.0 / 30.0) 
        
        # 1. Calculate Implicit Zoom
        # zoom = src_w / crop_w
        # This allows us to smooth the 'zoom' parameter in the filter
        current_zoom = self.src_w / max(1.0, float(crop_w))
        
        # 2. Update Filters (cx, cy, zoom)
        # Note: We pass current_zoom to the filter, which smooths it against previous zoom state
        cx, cy, smoothed_zoom = self.update(t, target_cx, target_cy, current_zoom)
        
        # 3. Recalculate Crop Dimensions from Smoothed Zoom
        # zoom = src_w / crop_w  =>  crop_w = src_w / zoom
        smoothed_zoom = max(1.0, smoothed_zoom)
        new_crop_w = int(self.src_w / smoothed_zoom)
        new_crop_h = int(self.src_h / smoothed_zoom) # Assuming square pixels/uniform zoom
        
        # Maintain aspect if original crop_h was different ratio?
        # Actually in vertical video, aspect ratio is fixed.
        # But let's be safe: scale h by same factor as w
        scale_factor = new_crop_w / crop_w if crop_w > 0 else 1.0
        new_crop_h = int(crop_h * scale_factor)
        
        # 4. Calculate top-left from center
        x = int(cx - (new_crop_w / 2))
        y = int(cy - (new_crop_h / 2))
        
        # 5. Clamp to bounds
        max_x = self.src_w - new_crop_w
        max_y = self.src_h - new_crop_h
        
        # Ensure crop size is valid (<= src)
        if max_x < 0: 
            new_crop_w = self.src_w
            x = 0
            max_x = 0
        if max_y < 0:
            new_crop_h = self.src_h
            y = 0
            max_y = 0
            
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        
        return x, y, new_crop_w, new_crop_h


class SceneChangeDetector:
    """
    Detects scene changes (cuts) using histogram difference.
    On a cut, the camera should instantly reset instead of smoothly panning.
    """
    
    def __init__(self, threshold: float = 0.35):
        """
        Args:
            threshold: Histogram difference threshold (0-1). Higher = less sensitive.
                       0.35 works well for typical edits.
        """
        self.threshold = threshold
        self.prev_hist = None
    
    def is_scene_change(self, frame_tensor: torch.Tensor) -> bool:
        """
        Check if current frame is a scene change from previous.
        
        Args:
            frame_tensor: [C, H, W] or [B, C, H, W] tensor in 0-1 range
        
        Returns:
            True if scene change detected
        """
        # Handle batch dimension
        if frame_tensor.dim() == 4:
            frame_tensor = frame_tensor[0]
        
        # Convert to grayscale and compute histogram
        gray = frame_tensor.mean(dim=0)  # [H, W]
        
        # Quantize to bins for histogram
        bins = 32
        hist = torch.histc(gray, bins=bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)  # Normalize
        
        if self.prev_hist is None:
            self.prev_hist = hist
            return False
        
        # Compute histogram difference (L1 distance)
        diff = (hist - self.prev_hist).abs().sum().item()
        
        self.prev_hist = hist
        
        return diff > self.threshold
    
    def reset(self):
        """Reset detector state (for new video)."""
        self.prev_hist = None

class ZeroCopyPipeline:
    """
    Production-quality GPU video processing pipeline with dynamic tracking.
    
    Features:
    - NVDEC hardware decoding
    - Per-frame Kalman-filtered tracking with deadband
    - Kornia GPU transforms (crop, resize)
    - NVENC hardware encoding
    - Audio passthrough with sync
    """
    
    def __init__(
        self,
        target_width: int = 1080,
        target_height: int = 1920,
        batch_size: int = 50,  # Process this many frames at once
        device: str = 'cuda',
        kalman_process_noise: float = 0.01,  # Kept for compat but unused
        kalman_measurement_noise: float = 10.0,
        kalman_deadband_pct: float = 0.15,
        one_euro_beta: float = 0.002,       # Ultra-low = no snapping (v2)
        one_euro_min_cutoff: float = 0.005,  # Ultra-low = very smooth base (v2)
        deadband_pct: float = 0.05,         # 5% position deadband to suppress micro-jitter
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.batch_size = batch_size
        self.device = device
        
        # One Euro parameters (v2: much smoother for multi-person)
        self.beta = one_euro_beta
        self.min_cutoff = one_euro_min_cutoff
        self.deadband_pct = deadband_pct
        
        print(f"[ZeroCopy] Initialized: {target_width}x{target_height}, batch={batch_size}, deadband={deadband_pct}")
    
    def reframe_with_tracking(
        self,
        video_path: str,
        tracking_data: Dict,
        output_path: str,
        transcript: Optional[Dict] = None,
        caption_engine: Optional[object] = None,
        pacing_plan: Optional[List[Dict]] = None,
        effects_plan: Optional[object] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Reframe video to vertical with per-frame dynamic tracking, pacing (zoom), and effects.
        
        Args:
            video_path: Input video path
            tracking_data: Dict with 'tracked_objects' containing face trajectories
            output_path: Output video path
            pacing_plan: List of interventions (zoom, etc.) from Boredom Detector
            effects_plan: EffectPlan from AudioAnalyzer (shake/flash triggers)
            progress_callback: Optional callback(current_frame, total_frames)
            
        Returns:
            True if successful
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        # Helper for resizing
        def resize_frames(tensor, size):
            try:
                import kornia
                return kornia.geometry.transform.resize(
                    tensor, size, interpolation='bilinear', antialias=True
                )
            except Exception:
                # Fallback to torch.nn.functional.interpolate
                import torch.nn.functional as F
                return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
        
        print(f"[ZeroCopy] Processing: {video_path.name}")
        
        # Get video info
        info = get_video_info(str(video_path))
        src_w, src_h = info['width'], info['height']
        fps = info['fps']
        fps_str = info['fps_str']
        duration = info['duration']
        total_frames = int(duration * fps)
        
        print(f"[ZeroCopy] Source: {src_w}x{src_h} @ {fps:.2f} FPS, ~{total_frames} frames")
        
        # Calculate crop dimensions (maintain aspect ratio for vertical)
        target_aspect = self.target_height / self.target_width  # 1920/1080 = 1.78
        crop_h = src_h
        crop_w = int(crop_h / target_aspect)
        
        if crop_w > src_w:
            crop_w = src_w
            crop_h = int(crop_w * target_aspect)
        
        print(f"[ZeroCopy] Crop window: {crop_w}x{crop_h}")
        
        # Extract trajectory data
        trajectory = self._extract_trajectory(tracking_data)
        traj_map = {p['frame_idx']: p for p in trajectory} if trajectory else {}
        
        print(f"[ZeroCopy] Trajectory points: {len(trajectory)}")
        if traj_map:
            min_idx = min(traj_map.keys())
            max_idx = max(traj_map.keys())
            # print(f"[ZeroCopy] DEBUG: traj_map frame range: {min_idx} - {max_idx}")
        
        # Build Zoom Map from Pacing Plan
        # Map frame_idx -> zoom_factor (float)
        zoom_map = {}
        if pacing_plan:
            print(f"[ZeroCopy] Applying {len(pacing_plan)} pacing interventions...")
            for action in pacing_plan:
                if action.get('type') == 'zoom':
                    start_f = int(action['start_time'] * fps)
                    end_f = int(action['end_time'] * fps)
                    zoom = float(action.get('value', 1.0))
                    
                    for f in range(start_f, end_f + 1):
                        zoom_map[f] = zoom
            print(f"[ZeroCopy] Generated zoom map for {len(zoom_map)} frames")
        
        # Build Effects Map from EffectPlan (shake/flash triggers)
        # Map frame_idx -> List of effect triggers
        effects_map = {}
        if effects_plan and hasattr(effects_plan, 'triggers'):
            print(f"[ZeroCopy] Processing {len(effects_plan.triggers)} effect triggers...")
            for trigger in effects_plan.triggers:
                frame_idx = int(trigger.timestamp * fps)
                duration_frames = max(1, int(trigger.duration * fps))
                
                for f in range(frame_idx, frame_idx + duration_frames):
                    if f not in effects_map:
                        effects_map[f] = []
                    effects_map[f].append({
                        'type': trigger.effect_type,
                        'intensity': trigger.intensity,
                    })
            print(f"[ZeroCopy] Effects scheduled for {len(effects_map)} frames")
        
        # Initialize OneEuroCamera
        camera = OneEuroCamera(
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            min_cutoff=self.min_cutoff,
            beta=self.beta
        )
        
        # Initialize Scene Change Detector for handling fast cuts
        scene_detector = SceneChangeDetector(threshold=0.35)
        
        # Create temp directory for output frames
        temp_dir = output_path.parent / f"_zerocopy_temp_{output_path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # ========== STAGE 1: DECODE + CROP + ENCODE (BATCHED) ==========
            print(f"[ZeroCopy] Starting batched processing (Zero-Copy Native)...")
            from .gpu_video_utils import decode_video_native_stream
            
            processed_frames = 0
            batch_tensors = []
            
            # Track last known target position for fallback when tracking is lost
            # STRATEGY: Look Ahead Initialization
            # Instead of initializing at center (which causes panning if detection is missed in first frames),
            # we find the FIRST valid tracking point in the future and initialize there.
            
            last_target_cx = src_w // 2
            last_target_cy = src_h // 2
            
            if traj_map:
                try:
                    # Find earliest frame with VALID tracking data
                    # (Skip frames with garbage coordinates like INT64_MIN)
                    sorted_frames = sorted(traj_map.keys())
                    
                    found_valid = False
                    
                    for first_idx in sorted_frames:
                        pt = traj_map[first_idx]
                        
                        # Validate coordinates Sanity Check
                        # cx should be within 0 and src_w (plus some margin for out of frame tracking)
                        # INT64_MIN is definitely invalid
                        if pt['cx'] < -1000 or pt['cx'] > src_w + 1000:
                             continue
                             
                        # Calculate Target (Logic duplicated from loop for consistency)
                        eye_y = None
                        if 'pose' in pt and pt['pose']:
                            pose = pt['pose']
                            if 'left_eye' in pose and 'right_eye' in pose:
                                eye_y = (pose['left_eye']['y'] + pose['right_eye']['y']) / 2
                            elif 'nose' in pose:
                                eye_y = pose['nose']['y']
                                
                        if eye_y is None:
                            if pt.get('label') == 'person':
                                 eye_y = pt['cy'] - (pt['h'] * 0.35)
                            else:
                                 eye_y = pt['cy'] - (pt['h'] * 0.2)
                        
                        cinematic_offset = int(crop_h * 0.17)
                        last_target_cy = int(eye_y + cinematic_offset)
                        last_target_cx = pt['cx']
                        
                        print(f"[ZeroCopy] Look-Ahead Init: Set initial target to ({last_target_cx}, {last_target_cy}) from frame {first_idx}")
                        found_valid = True
                        break
                    
                    if not found_valid:
                        print(f"[ZeroCopy] Warning: Look-Ahead Init could not find ANY valid points in trajectory. Using center.")

                except Exception as e:
                    print(f"[ZeroCopy] Warning: Look-Ahead Init failed: {e}")
                    # Fallback to center remains

            
            # Use native generator
            # frame_gen yields (tensor, info) where tensor is on GPU
            frame_gen = decode_video_native_stream(str(video_path), device=self.device)
            
            for frame_idx, (tensor, _) in enumerate(frame_gen):
                # tensor is (C, H, W) float32 on GPU
                
                # === SCENE CHANGE DETECTION ===
                # If a cut is detected, we need to reset the camera instantly
                # to avoid the smooth "pan across screen" jitter
                is_cut = scene_detector.is_scene_change(tensor)
                
                # Get target position for this frame
                if frame_idx in traj_map:
                    pt = traj_map[frame_idx]
                    
                    # === Cinematic Targeting (Rule of Thirds) ===
                    # Goal: Subject eyes at 33% of frame height (Top Third line)
                    # KalmanCamera targets CENTER (50%).
                    # Offset needed: Target = EyeY + (0.5 - 0.33) * CropH
                    # Target = EyeY + 0.17 * CropH
                    
                    eye_y = None
                    
                    # Try Pose Keypoints (Best)
                    if 'pose' in pt and pt['pose']:
                        pose = pt['pose']
                        if 'left_eye' in pose and 'right_eye' in pose:
                            eye_y = (pose['left_eye']['y'] + pose['right_eye']['y']) / 2
                        elif 'nose' in pose:
                            eye_y = pose['nose']['y']
                            
                    # Fallback to Estimates
                    if eye_y is None:
                        # Estimate Face Center
                        # If detecting 'person', face is roughly top 20%
                        # cy is person center.
                        if pt.get('label') == 'person':
                             face_y_est = pt['cy'] - (pt['h'] * 0.35)
                             eye_y = face_y_est
                        else:
                             # It's a face/head bbox
                             # Eyes are roughly 30% down from top
                             eye_y = pt['cy'] - (pt['h'] * 0.2)
                    
                    # Apply Cinematic Offset
                    # We want eye_y to be at 33% of crop_h
                    # Calculate target_cy such that Camera Centers it.
                    # If Camera Centers Target, Target is at 50% Crop.
                    # We want Eye at 33%.
                    # Target - Eye = 17% Crop.
                    # Target = Eye + 17% Crop.
                    
                    cinematic_offset = int(crop_h * 0.17)
                    target_cy = int(eye_y + cinematic_offset)
                    target_cx = pt['cx'] # Keep X centered on subject
                    
                    # === POSITION DEADBAND ===
                    # If the new target is within deadband_pct of the current position,
                    # don't update. This prevents micro-jitter from noisy detections.
                    deadband_x = int(src_w * self.deadband_pct)
                    deadband_y = int(src_h * self.deadband_pct)
                    
                    if (abs(target_cx - last_target_cx) < deadband_x and
                        abs(target_cy - last_target_cy) < deadband_y):
                        # Within deadband — suppress update, hold position
                        target_cx = last_target_cx
                        target_cy = last_target_cy
                    else:
                        # Significant movement — update
                        last_target_cx = target_cx
                        last_target_cy = target_cy
                    
                else:
                    # HOLD last known position instead of jumping to center
                    # This prevents "panning away" when tracking is momentarily lost
                    target_cx = last_target_cx
                    target_cy = last_target_cy
                
                # Get smoothed crop position from Kalman camera
                # Apply Dynamic Zoom
                current_zoom = zoom_map.get(frame_idx, 1.0)
                current_crop_w = int(crop_w / current_zoom)
                current_crop_h = int(crop_h / current_zoom)
                
                # === CAMERA RESET ON SCENE CUT ===
                # If a cut was detected, reset the camera instantly to current target
                # This prevents the smooth "pan across screen" jitter
                if is_cut and frame_idx > 0:
                    camera.reset(target_cx, target_cy)
                
                crop_x, crop_y, smooth_crop_w, smooth_crop_h = camera.get_crop(target_cx, target_cy, current_crop_w, current_crop_h)
                
                # Crop on GPU (Zero-Copy)
                # Kornia/Torch slicing works on VRAM
                # Ensure crop fits (clamp is already done in get_crop but double safety)
                
                # Crop on GPU (Zero-Copy)
                cropped = tensor[:, crop_y:crop_y+smooth_crop_h, crop_x:crop_x+smooth_crop_w]
                
                # Must resize HERE if zoom changes crop size, otherwise stack fails
                # We lose batch resize efficiency, but stacking variable sizes is impossible.
                resized_frame = resize_frames(
                    cropped.unsqueeze(0),
                    (self.target_height, self.target_width)
                ).squeeze(0)
                
                # === APPLY EFFECTS (Shake/Flash/Zoom Punch) ===
                if frame_idx in effects_map:
                    import random
                    for effect in effects_map[frame_idx]:
                        if effect['type'] == 'shake':
                            # Camera shake: random offset translation (v2: stronger)
                            intensity = effect['intensity']
                            offset_px = int(intensity * 50)  # Max 50px shift (was 20)
                            offset_x = random.randint(-offset_px, offset_px)
                            offset_y = random.randint(-offset_px, offset_px)
                            # Apply via roll (circular shift - fast on GPU)
                            resized_frame = torch.roll(resized_frame, shifts=(offset_y, offset_x), dims=(1, 2))
                        elif effect['type'] == 'flash':
                            # White flash: brightness boost (v2: more visible)
                            intensity = min(effect['intensity'], 0.35)  # Cap at 35%
                            resized_frame = torch.clamp(resized_frame + intensity, 0.0, 1.0)
                        elif effect['type'] == 'zoom_punch':
                            # Zoom punch: scale up center by 1.05-1.15x for impact
                            zoom_factor = 1.0 + (effect['intensity'] * 0.15)  # 1.05-1.15x
                            _, fh, fw = resized_frame.shape
                            crop_h_z = int(fh / zoom_factor)
                            crop_w_z = int(fw / zoom_factor)
                            y_off = (fh - crop_h_z) // 2
                            x_off = (fw - crop_w_z) // 2
                            zoomed = resized_frame[:, y_off:y_off+crop_h_z, x_off:x_off+crop_w_z]
                            resized_frame = resize_frames(zoomed.unsqueeze(0), (fh, fw)).squeeze(0)
                        # Map 'beat_drop' and 'energy_peak' to zoom_punch
                        elif effect['type'] in ('beat_drop', 'energy_peak', 'impact'):
                            # Treat high-energy audio triggers as zoom punches
                            zoom_factor = 1.0 + (effect['intensity'] * 0.10)  # 1.05-1.10x
                            _, fh, fw = resized_frame.shape
                            crop_h_z = int(fh / zoom_factor)
                            crop_w_z = int(fw / zoom_factor)
                            y_off = (fh - crop_h_z) // 2
                            x_off = (fw - crop_w_z) // 2
                            zoomed = resized_frame[:, y_off:y_off+crop_h_z, x_off:x_off+crop_w_z]
                            resized_frame = resize_frames(zoomed.unsqueeze(0), (fh, fw)).squeeze(0)
                
                batch_tensors.append(resized_frame)
                
                # Process batch when full
                if len(batch_tensors) >= self.batch_size:
                    batch = torch.stack(batch_tensors) # Already on device
                    
                    # Batch is already resized
                    resized = batch
                    
                    # Apply GPU Captions (Zero-Copy Overlay)
                    if caption_engine and transcript:
                        # Process batch for captions
                        # Timestamps for batch
                        current_fps = fps
                        batch_start_idx = processed_frames
                        
                        for b_i in range(resized.shape[0]):
                           ts = (batch_start_idx + b_i) / current_fps
                           
                           # Render caption tensor (4, H, W)
                           cap_tensor = caption_engine.render_to_tensor(
                               transcript.get("words", []),
                               ts,
                               self.target_width,
                               self.target_height
                           )
                           
                           if cap_tensor is not None:
                               # Composite: Out = Frame * (1 - Alpha) + Cap * Alpha
                               # Frame is (3, H, W), Cap is (4, H, W)
                               
                               alpha = cap_tensor[3:4, :, :]
                               rgb = cap_tensor[0:3, :, :]
                               
                               # In-place update to save memory
                               resized[b_i] = resized[b_i] * (1.0 - alpha) + rgb * alpha
                               
                               del cap_tensor
                    
                    # Save batch frames
                    self._save_batch(resized, temp_dir, processed_frames)
                    
                    processed_frames += len(batch_tensors)
                    batch_tensors.clear()  # Release individual tensor refs before cache clear
                    
                    # Clear GPU cache
                    del batch, resized
                    clear_gpu_cache()
                    
                    if progress_callback:
                        progress_callback(processed_frames, total_frames)
                    
                    if processed_frames % 200 == 0:
                        print(f"[ZeroCopy] Processed {processed_frames}/{total_frames}...")
            
            # Close generator implicit by loop exit
            # process logic removed as generator handles cleanup
            
            # Process remaining frames
            if batch_tensors:
                batch = torch.stack(batch_tensors)
                # Frames are already resized per-frame (due to zoom), skip batch resize
                resized = batch
                
                # Apply GPU Captions (Residual Batch)
                if caption_engine and transcript:
                     batch_start_idx = processed_frames
                     for b_i in range(resized.shape[0]):
                        ts = (batch_start_idx + b_i) / fps
                        cap_tensor = caption_engine.render_to_tensor(
                            transcript.get("words", []),
                            ts,
                            self.target_width,
                            self.target_height
                        )
                        if cap_tensor is not None:
                            alpha = cap_tensor[3:4]
                            rgb = cap_tensor[0:3]
                            resized[b_i] = resized[b_i] * (1.0 - alpha) + rgb * alpha
                            del cap_tensor

                self._save_batch(resized, temp_dir, processed_frames)
                processed_frames += len(batch_tensors)
                batch_tensors.clear()  # Release individual tensor refs
                del batch, resized
                clear_gpu_cache()
            
            print(f"[ZeroCopy] Processed {processed_frames} total frames")
            
            # ========== STAGE 2: ENCODE WITH AUDIO ==========
            print(f"[ZeroCopy] Encoding with NVENC...")
            
            encode_cmd = [
                'ffmpeg', '-y',
                '-framerate', fps_str,
                '-i', str(temp_dir / 'frame_%06d.jpg'),
                # Audio input with perceptual offset (-50ms = audio plays 50ms early)
                '-itsoffset', '-0.05',
                '-i', str(video_path),
                '-map', '0:v',
                '-map', '1:a?',
                # A/V Sync (v2: robust aresample instead of basic -async)
                '-avoid_negative_ts', 'make_zero',
                '-vsync', 'cfr',
                '-shortest',
                # Force 1080p30 output per spec (F1.1, F1.2)
                '-r', '30',
            ]
            
            # Add encoding flags based on device
            if self.device == 'cuda':
                encode_cmd.extend([
                    '-c:v', 'h264_nvenc',       # h264 for universal playback
                    '-preset', 'p5',            # Quality preset (p5 = high quality)
                    '-rc', 'vbr',               # Variable bitrate
                    '-b:v', '12M',              # 12Mbps for crisp 1080x1920 vertical
                    '-maxrate', '15M',          # Peak bitrate cap
                    '-bufsize', '24M',          # VBV buffer
                    '-pix_fmt', 'yuv420p',      # Universal pixel format
                    '-profile:v', 'high',       # H.264 High profile
                ])
            else:
                encode_cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                ])
            
            # Audio settings (v2: aresample for robust sync)
            encode_cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-af', 'aresample=async=1:first_pts=0',
                str(output_path)
            ])
            
            result = subprocess.run(encode_cmd, capture_output=True)
            
            # Print FFmpeg output for debugging
            if result.returncode != 0 or not output_path.exists():
                print(f"[ZeroCopy] FFmpeg stderr:\n{result.stderr.decode()}")
                
            if result.returncode != 0:
                print(f"[ZeroCopy] Encode failed: {result.stderr.decode()[:500]}")
                return False
            
            print(f"[ZeroCopy] Success! Output: {output_path}")
            return True
            
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def _extract_trajectory(self, tracking_data: Dict) -> List[Dict]:
        """Extract the best face/person trajectory from tracking data."""
        if not tracking_data:
            print("[ZeroCopy] No tracking data provided")
            return []
        
        # Debug: Print what we received
        tracked_objs = tracking_data.get('tracked_objects', [])
        # print(f"[ZeroCopy] DEBUG: Received {len(tracked_objs)} tracked objects")
        
        best_traj = []
        max_score = -1.0
        best_label = ""
        
        for obj in tracking_data.get('tracked_objects', []):
            label = obj.get('label', '')
            if label in ('face', 'person') and 'trajectory' in obj:
                traj = obj['trajectory']
                num_points = len(traj)
                
                # Calculate average area
                total_area = sum(p['w'] * p['h'] for p in traj)
                avg_area = total_area / num_points if num_points > 0 else 0
                
                # Simple score: biased towards duration AND size (foreground speaker)
                score = num_points + (avg_area / 1000.0)
                
                if score > max_score:
                    max_score = score
                    best_traj = traj
                    best_label = label
        
        if best_traj:
            print(f"[ZeroCopy] Found best trajectory '{best_label}' with score {max_score:.1f}")
            # Debug: Print first 5 frame indices
            # first_indices = [p.get('frame_idx', -1) for p in best_traj[:5]]
            # print(f"[ZeroCopy] DEBUG: First 5 frame indices: {first_indices}")
            return best_traj
        
        print(f"[ZeroCopy] No face/person trajectory found in tracking data")
        return []
    
    def _save_batch(self, batch: torch.Tensor, output_dir: Path, start_idx: int):
        """Save a batch of frames as JPEG files."""
        import cv2
        
        batch_cpu = (batch.cpu() * 255).byte()
        
        for i, frame in enumerate(batch_cpu):
            # CHW -> HWC, RGB -> BGR
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            output_path = output_dir / f"frame_{start_idx + i:06d}.jpg"
            cv2.imwrite(str(output_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def reframe_zero_copy(
    video_path: str,
    tracking_data: Dict,
    output_path: str,
    target_width: int = 1080,
    target_height: int = 1920,
    transcript: Optional[Dict] = None,
    caption_engine: Optional[object] = None,
    pacing_plan: Optional[List[Dict]] = None,
    effects_plan: Optional[object] = None,
    progress_callback: Optional[callable] = None,
    **pipeline_kwargs
) -> bool:
    """
    Convenience function for zero-copy reframing.
    
    Args:
        video_path: Input video path
        tracking_data: Tracking data with face trajectories
        output_path: Output video path
        target_width: Output width (default 1080)
        target_height: Output height (default 1920)
        **pipeline_kwargs: Additional ZeroCopyPipeline constructor arguments
        
    Returns:
        True if successful
    """
    pipeline = ZeroCopyPipeline(
        target_width=target_width,
        target_height=target_height,
        **pipeline_kwargs
    )
    
    return pipeline.reframe_with_tracking(
        video_path,
        tracking_data,
        output_path,
        transcript=transcript,
        caption_engine=caption_engine,
        pacing_plan=pacing_plan,
        effects_plan=effects_plan,
        progress_callback=progress_callback,
    )

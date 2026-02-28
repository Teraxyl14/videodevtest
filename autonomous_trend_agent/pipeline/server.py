"""
AI Video Pipeline — Dashboard Server (Hardened)
FastAPI + WebSocket server for real-time pipeline control and monitoring.

Crash protections:
  1. Thread crash recovery — auto-reset state if thread dies
  2. VRAM OOM catching — specific CUDA OOM handling with cleanup
  3. File validation — size >0, valid extension, readable before starting
  4. Stale state watchdog — periodic check if thread died silently
  5. Graceful shutdown — signal handlers + atexit for clean exit
  6. Race condition guard — threading.Lock for pipeline_orchestrator
  7. Dead WebSocket cleanup — removal on broadcast failure
  8. Video corruption check — ffprobe validation before pipeline
  9. Disk space check — verify output dir has room before starting
 10. Pipeline isolation — each run gets fresh orchestrator instance

Endpoints:
  GET  /api/status          — Pipeline state + system stats
  GET  /api/videos          — List available videos
  GET  /api/outputs         — List completed outputs (categorized)
  POST /api/run             — Start pipeline
  POST /api/stop            — Kill switch (immediate stop)
  POST /api/pause           — Pause/resume pipeline
  GET  /api/system          — GPU/CPU/RAM utilization
  WS   /ws                  — Real-time progress stream

Run inside Docker:
  python -m autonomous_trend_agent.pipeline.server
"""

import os
import sys
import json
import time
import signal
import asyncio
import atexit
import shutil
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import asdict

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Pipeline
from .orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResult, run_pipeline
from .ws_callback import WebSocketCallback


# ============================================================================
# App Setup
# ============================================================================
app = FastAPI(title="AI Video Pipeline Dashboard", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
ws_callback = WebSocketCallback()
pipeline_thread: Optional[threading.Thread] = None
pipeline_orchestrator: Optional[PipelineOrchestrator] = None
pipeline_result: Optional[PipelineResult] = None
cancel_flag = threading.Event()
pause_flag = threading.Event()
pipeline_lock = threading.Lock()  # Guard against race conditions

# Directories (inside Docker container)
VIDEOS_DIR = Path(os.getenv("INPUT_DIR", "/app/downloaded_videos"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))

# Valid video extensions
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

# Minimum disk space required (500MB)
MIN_DISK_SPACE_MB = 500

# Minimum valid video file size (10KB — anything smaller is likely corrupted)
MIN_VIDEO_SIZE_BYTES = 10240


# ============================================================================
# Request/Response Models
# ============================================================================
class RunRequest(BaseModel):
    video_path: str
    num_shorts: int = 4
    use_captions: bool = True
    use_effects: bool = True
    use_tracking: bool = True
    use_analysis: bool = True


class ConfigUpdate(BaseModel):
    num_shorts: Optional[int] = None
    use_captions: Optional[bool] = None
    use_effects: Optional[bool] = None
    use_tracking: Optional[bool] = None
    use_analysis: Optional[bool] = None
    caption_style: Optional[str] = None


# ============================================================================
# Validation Helpers
# ============================================================================
def validate_video_file(path: Path) -> Optional[str]:
    """
    Validate a video file before processing.
    Returns error message if invalid, None if OK.
    """
    if not path.exists():
        return f"File not found: {path}"
    
    if not path.is_file():
        return f"Not a file: {path}"
    
    if path.suffix.lower() not in VIDEO_EXTS:
        return f"Unsupported format: {path.suffix} (expected: {', '.join(VIDEO_EXTS)})"
    
    stat = path.stat()
    if stat.st_size < MIN_VIDEO_SIZE_BYTES:
        return f"File too small ({stat.st_size} bytes) — likely corrupted: {path.name}"
    
    # Try to read the first few bytes to verify it's accessible
    try:
        with open(path, 'rb') as f:
            header = f.read(16)
            if len(header) < 16:
                return f"File unreadable or truncated: {path.name}"
    except PermissionError:
        return f"Permission denied: {path.name}"
    except OSError as e:
        return f"Cannot read file: {path.name} ({e})"
    
    return None  # Valid


def validate_video_ffprobe(path: Path) -> Optional[str]:
    """
    Quick ffprobe check to verify the file is a valid video container.
    Returns error message if invalid, None if OK.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type,duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:200]
            return f"FFprobe rejected: {path.name} — {stderr}"
        
        output = result.stdout.strip()
        if not output or "video" not in output:
            return f"No video stream found in: {path.name}"
        
        return None  # Valid
    except FileNotFoundError:
        # ffprobe not installed — skip this check
        return None
    except subprocess.TimeoutExpired:
        return f"FFprobe timed out for: {path.name}"
    except Exception as e:
        # Don't block pipeline for ffprobe failures
        return None


def check_disk_space(path: Path) -> Optional[str]:
    """Check if there's enough disk space for output."""
    try:
        usage = shutil.disk_usage(str(path))
        free_mb = usage.free / (1024 * 1024)
        if free_mb < MIN_DISK_SPACE_MB:
            return f"Low disk space: {free_mb:.0f}MB free (need {MIN_DISK_SPACE_MB}MB)"
    except Exception:
        pass  # Don't block on disk check failures
    return None


def free_gpu_memory():
    """Force-free GPU memory after a crash."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ============================================================================
# System Monitoring
# ============================================================================
def get_system_stats() -> dict:
    """Get real-time GPU/CPU/RAM utilization."""
    stats = {
        "gpu": None,
        "cpu_pct": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_pct": None,
    }
    
    # GPU stats via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 5:
                stats["gpu"] = {
                    "utilization_pct": int(parts[0].strip()),
                    "vram_used_mb": int(parts[1].strip()),
                    "vram_total_mb": int(parts[2].strip()),
                    "vram_pct": round(int(parts[1].strip()) / max(1, int(parts[2].strip())) * 100, 1),
                    "temp_c": int(parts[3].strip()),
                    "name": parts[4].strip(),
                }
    except Exception:
        pass
    
    # CPU + RAM via /proc
    try:
        with open("/proc/stat") as f:
            line = f.readline()
            parts = line.split()
            idle = int(parts[4])
            total = sum(int(p) for p in parts[1:])
            stats["cpu_pct"] = round(100.0 * (1 - idle / max(1, total)), 1)
    except Exception:
        pass
    
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                mem[parts[0].rstrip(":")] = int(parts[1])
            total_kb = mem.get("MemTotal", 0)
            avail_kb = mem.get("MemAvailable", 0)
            used_kb = total_kb - avail_kb
            stats["ram_total_gb"] = round(total_kb / 1048576, 1)
            stats["ram_used_gb"] = round(used_kb / 1048576, 1)
            stats["ram_pct"] = round(used_kb / max(1, total_kb) * 100, 1)
    except Exception:
        pass
    
    return stats


# ============================================================================
# Pipeline Runner (runs in background thread)
# ============================================================================
def _safe_shutdown_orchestrator():
    """Safely shutdown the orchestrator with error handling."""
    global pipeline_orchestrator
    with pipeline_lock:
        if pipeline_orchestrator:
            try:
                pipeline_orchestrator.shutdown()
            except Exception as e:
                print(f"[Server] Orchestrator shutdown error: {e}")
            finally:
                pipeline_orchestrator = None
    free_gpu_memory()


def _process_single_video(video_path: str, config: PipelineConfig) -> Optional[PipelineResult]:
    """Process a single video with full error handling. Returns PipelineResult or None."""
    global pipeline_orchestrator
    
    path = Path(video_path)
    
    # Pre-flight validation
    err = validate_video_file(path)
    if err:
        ws_callback.on_error("Validation", err)
        return PipelineResult(
            success=False, video_path=video_path,
            output_dir=config.output_dir, shorts_created=[],
            duration_seconds=0, errors=[err]
        )
    
    ffprobe_err = validate_video_ffprobe(path)
    if ffprobe_err:
        ws_callback.on_error("Validation", ffprobe_err)
        return PipelineResult(
            success=False, video_path=video_path,
            output_dir=config.output_dir, shorts_created=[],
            duration_seconds=0, errors=[ffprobe_err]
        )
    
    try:
        with pipeline_lock:
            pipeline_orchestrator = PipelineOrchestrator(config=config, callback=ws_callback)
        
        result = pipeline_orchestrator.run(video_path)
        return result
    
    except MemoryError:
        error_msg = f"VRAM/RAM out of memory processing {path.name}"
        ws_callback.on_error("Memory", error_msg)
        free_gpu_memory()
        return PipelineResult(
            success=False, video_path=video_path,
            output_dir=config.output_dir, shorts_created=[],
            duration_seconds=0, errors=[error_msg]
        )
    
    except RuntimeError as e:
        error_str = str(e)
        if "CUDA" in error_str or "out of memory" in error_str.lower():
            error_msg = f"CUDA OOM: {path.name} — {error_str[:200]}"
            ws_callback.on_error("CUDA", error_msg)
            free_gpu_memory()
        else:
            error_msg = f"Runtime error: {path.name} — {error_str[:200]}"
            ws_callback.on_error("Pipeline", error_msg)
        return PipelineResult(
            success=False, video_path=video_path,
            output_dir=config.output_dir, shorts_created=[],
            duration_seconds=0, errors=[error_msg]
        )
    
    except Exception as e:
        error_msg = f"{path.name}: {str(e)[:300]}"
        ws_callback.on_error("Pipeline", error_msg)
        tb = traceback.format_exc()
        print(f"[Server] Pipeline exception:\n{tb}")
        return PipelineResult(
            success=False, video_path=video_path,
            output_dir=config.output_dir, shorts_created=[],
            duration_seconds=0, errors=[error_msg]
        )
    
    finally:
        _safe_shutdown_orchestrator()


def _run_pipeline_thread(video_path: str, config: PipelineConfig):
    """Run pipeline in a background thread. Handles both single files and directories."""
    global pipeline_result
    
    ws_callback.reset()
    cancel_flag.clear()
    pause_flag.clear()
    
    # Disk space check
    disk_err = check_disk_space(OUTPUT_DIR)
    if disk_err:
        ws_callback.on_error("System", disk_err)
        ws_callback.state = "error"
        return
    
    target = Path(video_path)
    
    try:
        if target.is_dir():
            # ---- BATCH MODE ----
            videos = sorted(
                f for f in target.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            )
            
            if not videos:
                ws_callback.on_error("Pipeline", f"No video files found in {video_path}")
                ws_callback.state = "error"
                return
            
            ws_callback.on_step(f"Batch mode: {len(videos)} video(s) found")
            
            all_results = []
            for i, vid in enumerate(videos):
                if cancel_flag.is_set():
                    ws_callback.state = "cancelled"
                    return
                
                ws_callback.on_step(
                    f"Processing video {i+1}/{len(videos)}: {vid.name}",
                    current=i+1, total=len(videos)
                )
                
                result = _process_single_video(str(vid), config)
                if result:
                    all_results.append(result)
            
            # Combine results
            success = any(r.success for r in all_results) if all_results else False
            pipeline_result = PipelineResult(
                success=success,
                video_path=video_path,
                output_dir=config.output_dir,
                shorts_created=[s for r in all_results for s in r.shorts_created],
                duration_seconds=sum(r.duration_seconds for r in all_results),
                errors=[e for r in all_results for e in r.errors],
            )
            ws_callback.state = "complete" if success else "error"
        
        else:
            # ---- SINGLE FILE MODE ----
            result = _process_single_video(video_path, config)
            pipeline_result = result
            
            if cancel_flag.is_set():
                ws_callback.state = "cancelled"
            elif result and result.success:
                ws_callback.state = "complete"
            else:
                ws_callback.state = "error"
    
    except Exception as e:
        # Catch-all: should never reach here, but guarantees we recover
        ws_callback.on_error("Server", f"Unexpected crash: {str(e)[:300]}")
        ws_callback.state = "error"
        traceback.print_exc()
        _safe_shutdown_orchestrator()


# ============================================================================
# Watchdog: detect zombie pipeline threads
# ============================================================================
async def _watchdog_task():
    """Periodically check if the pipeline thread died without cleanup."""
    while True:
        await asyncio.sleep(5)
        if pipeline_thread and not pipeline_thread.is_alive():
            if ws_callback.state == "running":
                ws_callback.on_error("Watchdog", "Pipeline thread died unexpectedly")
                ws_callback.state = "error"
                _safe_shutdown_orchestrator()


@app.on_event("startup")
async def start_watchdog():
    asyncio.create_task(_watchdog_task())


# ============================================================================
# Graceful Shutdown
# ============================================================================
def _shutdown_handler(signum=None, frame=None):
    """Clean shutdown on SIGTERM/SIGINT."""
    print("\n[Server] Shutting down gracefully...")
    cancel_flag.set()
    _safe_shutdown_orchestrator()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
atexit.register(lambda: _safe_shutdown_orchestrator())


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the dashboard."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>Dashboard files not found</h1>", status_code=404)


@app.get("/api/status")
async def get_status():
    """Get current pipeline state."""
    elapsed = time.time() - ws_callback._start_time if ws_callback._start_time else 0
    
    # Auto-detect zombie state
    is_running = pipeline_thread and pipeline_thread.is_alive()
    state = ws_callback.state
    if state == "running" and not is_running:
        state = "error"
    
    result_summary = None
    if pipeline_result:
        result_summary = {
            "success": pipeline_result.success,
            "shorts_created": len(pipeline_result.shorts_created),
            "duration": pipeline_result.duration_seconds,
            "errors": pipeline_result.errors,
        }
    
    return {
        "state": state,
        "current_stage": ws_callback._current_stage,
        "stage_progress": ws_callback._stage_progress,
        "completed_stages": list(ws_callback._completed_stages),
        "overall_pct": ws_callback._overall_pct,
        "elapsed_seconds": elapsed,
        "errors": ws_callback._errors,
        "result": result_summary,
    }


@app.get("/api/videos")
async def list_videos():
    """List available videos and directories."""
    videos = []
    dirs = []
    
    for search_dir in [VIDEOS_DIR, Path("/app/input")]:
        if not search_dir.exists():
            continue
        
        try:
            for item in sorted(search_dir.iterdir()):
                try:
                    if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
                        stat = item.stat()
                        videos.append({
                            "name": item.name,
                            "path": str(item),
                            "size_mb": round(stat.st_size / 1048576, 1),
                            "modified": stat.st_mtime,
                            "directory": str(search_dir),
                        })
                    elif item.is_dir():
                        sub_vids = sum(1 for f in item.iterdir() 
                                     if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
                        if sub_vids > 0:
                            dirs.append({
                                "name": item.name,
                                "path": str(item),
                                "video_count": sub_vids,
                            })
                except PermissionError:
                    continue  # Skip inaccessible files
        except PermissionError:
            continue  # Skip inaccessible directories
    
    return {"videos": videos, "directories": dirs}


@app.get("/api/outputs")
async def list_outputs():
    """List completed outputs, categorized by source video."""
    if not OUTPUT_DIR.exists():
        return {"outputs": []}
    
    outputs = []
    try:
        for item in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if not item.is_dir():
                continue
            
            shorts = []
            try:
                for short_dir in sorted(item.iterdir()):
                    if not short_dir.is_dir():
                        continue
                    
                    video_file = short_dir / "reframed.mp4"
                    if not video_file.exists():
                        video_file = next(short_dir.glob("*.mp4"), None)
                    
                    metadata_file = short_dir / "metadata.json"
                    metadata = {}
                    if metadata_file and metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text())
                        except Exception:
                            pass
                    
                    ai_file = short_dir / "ai_enhancement.json"
                    ai_data = {}
                    if ai_file.exists():
                        try:
                            ai_data = json.loads(ai_file.read_text())
                        except Exception:
                            pass
                    
                    shorts.append({
                        "name": short_dir.name,
                        "path": str(short_dir),
                        "has_video": video_file is not None and video_file.exists(),
                        "video_path": str(video_file) if video_file and video_file.exists() else None,
                        "metadata": metadata,
                        "ai_enhancement": ai_data,
                    })
            except PermissionError:
                pass  # Skip inaccessible output directories
            
            outputs.append({
                "name": item.name,
                "path": str(item),
                "shorts": shorts,
                "num_shorts": len(shorts),
                "created": item.stat().st_mtime if item.exists() else 0,
            })
    except PermissionError:
        pass
    
    return {"outputs": outputs}


@app.get("/api/system")
async def system_stats():
    """Real-time system utilization."""
    return get_system_stats()


@app.post("/api/run")
async def start_pipeline(req: RunRequest):
    """Start the pipeline on a video or directory."""
    global pipeline_thread
    
    # Check if already running
    if pipeline_thread and pipeline_thread.is_alive():
        raise HTTPException(400, "Pipeline already running. Stop it first.")
    
    # Validate path
    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(404, f"Path not found: {req.video_path}")
    
    # Single file validation
    if video_path.is_file():
        err = validate_video_file(video_path)
        if err:
            raise HTTPException(422, err)
    
    # Directory validation
    elif video_path.is_dir():
        videos_found = sum(
            1 for f in video_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        )
        if videos_found == 0:
            raise HTTPException(422, f"No video files found in {req.video_path}")
    
    # Disk space check
    disk_err = check_disk_space(OUTPUT_DIR)
    if disk_err:
        raise HTTPException(507, disk_err)
    
    config = PipelineConfig(
        output_dir=str(OUTPUT_DIR),
        num_shorts=req.num_shorts,
        use_captions=req.use_captions,
        use_effects=req.use_effects,
        use_yolo_tracking=req.use_tracking,
        use_qwen_analysis=req.use_analysis,
        device="cuda",
    )
    
    # Start in background thread
    pipeline_thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(str(video_path), config),
        daemon=True,
        name="pipeline-runner",
    )
    pipeline_thread.start()
    
    mode = "batch" if video_path.is_dir() else "single"
    return {"status": "started", "video_path": str(video_path), "mode": mode}


@app.post("/api/stop")
async def stop_pipeline():
    """Kill switch — immediately stop the pipeline."""
    cancel_flag.set()
    _safe_shutdown_orchestrator()
    ws_callback.state = "cancelled"
    return {"status": "stopped"}


@app.post("/api/pause")
async def toggle_pause():
    """Toggle pause/resume."""
    if pause_flag.is_set():
        pause_flag.clear()
        ws_callback.state = "running"
        return {"status": "resumed"}
    else:
        pause_flag.set()
        ws_callback.state = "paused"
        return {"status": "paused"}


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Real-time progress stream."""
    await ws.accept()
    ws_callback.set_loop(asyncio.get_event_loop())
    ws_callback.add_client(ws)
    
    # Send current state snapshot on connect
    try:
        await ws.send_json(ws_callback.get_snapshot())
    except Exception:
        ws_callback.remove_client(ws)
        return
    
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong", "timestamp": time.time()})
            elif data == "snapshot":
                await ws.send_json(ws_callback.get_snapshot())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass  # Handle any unexpected WebSocket errors
    finally:
        ws_callback.remove_client(ws)


# ============================================================================
# Static Files (frontend)
# ============================================================================
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("  AI VIDEO PIPELINE — DASHBOARD SERVER (v1.1)")
    print("=" * 60)
    print(f"  Frontend: http://localhost:8080")
    print(f"  API:      http://localhost:8080/api/status")
    print(f"  WebSocket: ws://localhost:8080/ws")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "autonomous_trend_agent.pipeline.server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )

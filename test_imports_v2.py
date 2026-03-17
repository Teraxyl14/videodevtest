import sys
from pathlib import Path
import traceback

# Add project root to sys.path
project_root = Path(r"m:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video")
sys.path.insert(0, str(project_root))

errors = []

def catch_import(module_name, import_stmt):
    try:
        exec(import_stmt)
        print(f"[OK] {module_name} imported successfully.")
    except Exception as e:
        errors.append(f"[ERROR] {module_name} import failed:\n{traceback.format_exc()}")

catch_import("Orchestrator", "from autonomous_trend_agent.pipeline.orchestrator import PipelineOrchestrator")
catch_import("IPC Utils", "from autonomous_trend_agent.core.ipc_utils import initialize_persistent_hub_buffer, ephemeral_spoke_ingestion_protocol")
catch_import("ZeroCopyPipeline", "from autonomous_trend_agent.editor.zero_copy_pipeline import ZeroCopyPipeline")
catch_import("GPU Video Utils", "from autonomous_trend_agent.editor.gpu_video_utils import decode_video_native_stream")
catch_import("WhisperX Transcriber", "from autonomous_trend_agent.audio.whisperx_transcriber import execute_audio_alignment_spoke")
catch_import("Qwen3VideoAnalyzer", "from autonomous_trend_agent.brain.qwen3_video_analyzer import Qwen3VideoAnalyzer")
catch_import("YOLOv11Tracker", "from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker")
catch_import("GeminiVideoAnalyzer", "from autonomous_trend_agent.brain.video_analyzer import GeminiVideoAnalyzer")

if errors:
    print("\n=== ERRORS FOUND ===")
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print("\n=== All imports successful! ===")

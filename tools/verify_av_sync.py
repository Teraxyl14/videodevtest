
import subprocess
import json
import sys
from pathlib import Path

def get_video_info(path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        audio_stream = next(s for s in data['streams'] if s['codec_type'] == 'audio')
        
        duration = float(data['format']['duration'])
        fps_str = video_stream['r_frame_rate']
        fps = eval(fps_str)
        frames = int(video_stream.get('nb_frames', 0))
        
        return {
            "duration": duration,
            "fps_str": fps_str,
            "fps": fps,
            "frames": frames,
            "audio_present": True
        }
    except Exception as e:
        return {"error": str(e)}

def verify_sync(input_path, output_path):
    print(f"Checking Input: {input_path}")
    in_info = get_video_info(input_path)
    print(json.dumps(in_info, indent=2))
    
    print(f"\nChecking Output: {output_path}")
    out_info = get_video_info(output_path)
    print(json.dumps(out_info, indent=2))
    
    # Tolerances
    duration_diff = abs(in_info["duration"] - out_info["duration"])
    fps_match = in_info["fps_str"] == out_info["fps_str"]
    
    print("\n" + "="*40)
    print("SYNC REPORT")
    print("="*40)
    print(f"Duration Diff: {duration_diff:.4f}s (Tolerance: 0.1s)")
    print(f"FPS Match:     {fps_match} ({in_info['fps_str']} vs {out_info['fps_str']})")
    
    if duration_diff < 0.1 and fps_match:
         print("\n✅ PASS: Audio/Video Alignment looks good.")
    else:
         print("\n❌ FAIL: Synchronization Drift Detected.")

if __name__ == "__main__":
    PROJECT_ROOT = Path("m:/Projects/AI_Video/AI_Video-20260116T154526Z-3-001/AI_Video")
    input_video = PROJECT_ROOT / "autonomous_trend_agent/downloaded_videos/test_clip_10s.mp4"
    output_video = PROJECT_ROOT / "output/shorts/test_clip_10s_final.mp4"
    
    if not input_video.exists() or not output_video.exists():
        print("ERROR: Files not found.")
        sys.exit(1)
        
    verify_sync(input_video, output_video)

from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker
import sys
import json

video_path = "autonomous_trend_agent/downloaded_videos/test_clip_10s.mp4"

print(f"Testing YOLO Tracking on: {video_path}")

try:
    tracker = YOLOv11Tracker(model_size="n", enable_pose=True)
    result = tracker.track_video(video_path, target_class="person")
    
    print(f"Status: {result['status']}")
    print(f"Frames processed: {result['processed_frames']}")
    print(f"Tracked objects: {len(result['tracked_objects'])}")
    if result['tracked_objects']:
        print(f"Trajectory length: {result['tracked_objects'][0]['frames_tracked']}")
        
    if result['processed_frames'] > 0:
        print("SUCCESS: Frames processed")
        sys.exit(0)
    else:
        print("FAILURE: No frames processed")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

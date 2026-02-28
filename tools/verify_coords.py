from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker
import sys
import json
import av

video_path = "autonomous_trend_agent/downloaded_videos/test_clip_10s.mp4"

print(f"Testing YOLO Tracking Coordinates on: {video_path}")

# Get video dimensions
container = av.open(video_path)
stream = container.streams.video[0]
width = stream.width
height = stream.height
container.close()
print(f"Video Dimensions: {width}x{height}")

tracker = YOLOv11Tracker(model_size="n", enable_pose=True)
try:
    result = tracker.track_video(video_path, target_class="person")
    
    if result['tracked_objects']:
        traj = result['tracked_objects'][0]['trajectory']
        print(f"Trajectory length: {len(traj)}")
        
        # Check first, middle, last
        samples = [traj[0], traj[len(traj)//2], traj[-1]]
        for i, pt in enumerate(samples):
            print(f"Sample {i}: Frame {pt['frame_idx']} - ({pt['cx']}, {pt['cy']}) Box: {pt['w']}x{pt['h']}")
            
            # Validate bounds
            if not (0 <= pt['cx'] <= width and 0 <= pt['cy'] <= height):
                print(f"❌ OUT OF BOUNDS!")
            else:
                print(f"✅ In bounds")
    else:
        print("No objects tracked")

except Exception as e:
    print(f"Error: {e}")

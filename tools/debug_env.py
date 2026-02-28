
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_env")

VIDEO_PATH = "/workspace/input/test_video.mp4"

def test_pynvvideocodec():
    logger.info("-" * 20)
    logger.info("Testing PyNvVideoCodec")
    try:
        import PyNvVideoCodec as nvc
        logger.info("Import successful")
        try:
            # Simple test to create decoder
            gpu_id = 0
            # Just try to demux to see if it reads the file
            demuxer = nvc.PyFFmpegDemuxer(VIDEO_PATH)
            logger.info(f"Demuxer created. Width: {demuxer.Width()}, Codec: {demuxer.Codec()}")
        except Exception as e:
            logger.error(f"PyNvVideoCodec runtime failure: {e}", exc_info=True)
    except ImportError:
        logger.error("PyNvVideoCodec NOT installed")

def test_opencv():
    logger.info("-" * 20)
    logger.info("Testing OpenCV")
    try:
        import cv2
        logger.info(f"OpenCV Version: {cv2.__version__}")
        
        # Check backends
        build_info = cv2.getBuildInformation()
        if "FFMPEG" in build_info:
            logger.info("OpenCV has FFMPEG support: YES")
        else:
            logger.warning("OpenCV has FFMPEG support: NO (This causes failures)")
            
        cap = cv2.VideoCapture(VIDEO_PATH)
        if cap.isOpened():
            logger.info(f"Successfully opened video via OpenCV")
            ret, frame = cap.read()
            if ret:
                 logger.info(f"Successfully read frame: {frame.shape}")
            else:
                 logger.error("Failed to read frame")
            cap.release()
        else:
            logger.error("OpenCV failed to open video (cap.isOpened() is False)")
            
    except ImportError:
        logger.error("opencv-python NOT installed")

if __name__ == "__main__":
    logger.info(f"Diagnosing Video Path: {VIDEO_PATH}")
    if Path(VIDEO_PATH).exists():
        logger.info("File exists.")
    else:
        logger.error("File DOES NOT exist (This is the problem)")
    
    test_pynvvideocodec()
    test_opencv()

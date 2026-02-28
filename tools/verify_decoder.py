
import sys
import logging
import time
from pathlib import Path
import torch

# Add project root to path
sys.path.append("/workspace")

from autonomous_trend_agent.core.gpu_pipeline import GPUVideoDecoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("verify_decoder")

def test_decoder(video_path: str):
    logger.info(f"Testing GPUVideoDecoder on: {video_path}")
    
    if not Path(video_path).exists():
        logger.error(f"File not found: {video_path}")
        return


    try:
        start_time = time.time()
        
        # Test 0: CUDA Check
        logger.info("Step 0: Checking CUDA...")
        if not torch.cuda.is_available():
            logger.error("CUDA not available in Torch!")
            return
        logger.info(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        
        # Test 1: Initialization (Opening file)
        logger.info("Step 1: Initializing Decoder...")
        decoder = GPUVideoDecoder(video_path, gpu_id=0)
        
        info = decoder.info
        logger.info(f"Success! Video Info: {info.width}x{info.height} @ {info.fps}fps, {info.duration}s")
        
        # Test 2: Reading Frames
        logger.info("Step 2: Reading first 100 frames...")
        frame_count = 0
        for frame in decoder:
            frame_count += 1
            if frame_count % 10 == 0:
                print(f".", end="", flush=True)
            if frame_count >= 100:
                break
        
        print("\n")
        logger.info(f"Success! Read {frame_count} frames.")
        
        # Test 3: cleanup
        logger.info("Step 3: Closing decoder...")
        decoder.close()
        logger.info("Success! Decoder closed.")
        
        elapsed = time.time() - start_time
        logger.info(f"Verification complete in {elapsed:.2f}s")
        
    except Exception as e:
        logger.error(f"Decoder Failed: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_decoder.py <video_path>")
        sys.exit(1)
    
    test_decoder(sys.argv[1])

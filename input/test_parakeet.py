import logging
import soundfile as sf
import numpy as np
import traceback
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add repo root to path so we can import autonomous_trend_agent
sys.path.append("/app")

try:
    from autonomous_trend_agent.audio.parakeet_transcriber import ParakeetTranscriber

    # Generate dummy audio
    sr = 16000
    # 5 seconds of silence/noise
    audio = np.random.uniform(-0.1, 0.1, sr*5)
    sf.write("test.wav", audio, sr)
    logger.info("Created test.wav")

    print("Instantiating ParakeetTranscriber(1.1b)...")
    pt = ParakeetTranscriber(model_size="1.1b")
    
    print("Transcribing video...")
    res = pt.transcribe("test.wav")
    print("Success:", res.text)
    
except Exception:
    traceback.print_exc()
finally:
    if os.path.exists("test.wav"):
        os.remove("test.wav")

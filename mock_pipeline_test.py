import os
import sys
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MockTest")

sys.path.append("/app")
output_dir = Path("/app/output/mock_test")
output_dir.mkdir(parents=True, exist_ok=True)
video_path = output_dir / "dummy_video.mp4"

logger.info("Generating dummy video bytes...")
with open(video_path, "wb") as f:
    f.write(b"0" * 2000)

from autonomous_trend_agent.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

class DummyViralMoment:
    def __init__(self, start_time, end_time, segments, hook):
        self.start_time = start_time
        self.end_time = end_time
        self.segments = segments
        self.hook = hook
        self.description = "mock description"
        self.score = 95.0
        self.explanation = "mock explanation"


def mock_subprocess_run(*args, **kwargs):
    cmd = args[0]
    if isinstance(cmd, list) and len(cmd) > 0 and 'ffmpeg' in str(cmd[0]).lower():
        logger.info(f"MOCK FFmpeg CMD: {' '.join(str(x) for x in cmd)}")
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"0" * 2000)
        res = MagicMock()
        res.returncode = 0
        return res
    
    # Just return success for anything else
    res = MagicMock()
    res.returncode = 0
    res.stdout = b""
    return res

def run_mock_test():
    logger.info("Configuring PipelineOrchestrator for mock setup...")
    config = PipelineConfig(
        video_path=str(video_path),
        output_dir=str(output_dir),
        trend_topic="Mock Test Topic",
        num_shorts=1,
        caption_style="tiktok",
        use_gpu_reframing=True,
        use_effects=False,
        use_captions=False,
        batch_size=50
    )

    orchestrator = PipelineOrchestrator(config=config)

    mock_transcript_data = {
        "words": [{"word": f"word{i} ", "start": float(i), "end": i + 0.9} for i in range(60)]
    }
    transcript_path = output_dir / "transcript.json"
    with open(transcript_path, "w") as f:
        json.dump(mock_transcript_data, f)
    config.transcript_path = str(transcript_path)

    mock_moment = DummyViralMoment(
        hook="Mock test viral hook over 30s",
        start_time=10.0,
        end_time=65.0,
        segments=[{"start": 10.0, "end": 65.0}]
    )
    class MockAnalyzerData:
        def __init__(self, moments):
            self.viral_moments = moments

    mock_analysis_result = MockAnalyzerData([mock_moment])
    class MockPipelineOrch(PipelineOrchestrator):
        def _run_analysis(self, *args, **kwargs):
            return mock_analysis_result
        def _run_tracking(self, *args, **kwargs):
            return {"tracked_objects": []}

    orchestrator._run_analysis = MockPipelineOrch._run_analysis.__get__(orchestrator)
    orchestrator._run_tracking = MockPipelineOrch._run_tracking.__get__(orchestrator)

    with patch('subprocess.run', side_effect=mock_subprocess_run):
        logger.info("Executing PipelineOrchestrator.run()...")
        result = orchestrator.run(str(video_path), str(transcript_path))
    
    logger.info("="*50)
    if result.success:
        logger.info(f"SUCCESS! Pipeline created {len(result.shorts_created)} short(s).")
        if result.shorts_created:
            short_dir = result.shorts_created[0]['short_dir']
            short_video = os.path.join(short_dir, "video.mp4")
            
            if os.path.exists(short_video) and os.path.getsize(short_video) > 1000:
                logger.info(f"✅ Verified output video generated at: {short_video}")
                logger.info("Mock Integration Test PASSED: All phases and FFmpeg logic succeeded!")
            else:
                logger.error(f"❌ output video.mp4 is MISSING or too small at {short_video}")
                sys.exit(1)
        else:
            logger.error("❌ Pipeline succeeded but no shorts were in the array.")
            sys.exit(1)
    else:
        logger.error(f"❌ PIPELINE FAILED with errors:\n{json.dumps(result.errors, indent=2)}")
        sys.exit(1)

if __name__ == '__main__':
    run_mock_test()

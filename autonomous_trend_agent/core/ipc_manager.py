import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, Literal
from dataclasses import dataclass, asdict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
JOBS_DIR = PROJECT_ROOT / "autonomous_trend_agent" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobRequest:
    job_type: str  # "analyze", "transcribe", "generate"
    payload: Dict[str, Any]
    job_id: str = ""
    status: str = JobStatus.PENDING
    created_at: float = 0.0

    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

@dataclass
class JobResult:
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: float = 0.0

class IPCManager:
    """
    Inter-Process Communication Manager using file-based JSON messaging.
    Allows isolated Conda environments to talk to the main controller.
    """
    def __init__(self, role: Literal["hub", "spoke_video", "spoke_audio"]):
        self.role = role
        self.jobs_dir = JOBS_DIR
    
    def submit_job(self, job_type: str, payload: Dict[str, Any]) -> str:
        """(Hub) Submit a job to the queue."""
        job = JobRequest(job_type=job_type, payload=payload)
        job_path = self.jobs_dir / f"{job.job_type}_{job.job_id}.json"
        
        with open(job_path, "w") as f:
            json.dump(asdict(job), f, indent=2)
            
        print(f"[IPC-Hub] Submitted job {job.job_id} ({job_type})")
        return job.job_id

    def get_result(self, job_id: str, timeout: int = 600) -> JobResult:
        """(Hub) Wait for a job result."""
        start_time = time.time()
        result_path = self.jobs_dir / f"result_{job_id}.json"
        
        while (time.time() - start_time) < timeout:
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                    return JobResult(**data)
                except Exception as e:
                    print(f"[IPC-Hub] Error reading result: {e}")
                    time.sleep(1)
            time.sleep(1)
            
        return JobResult(job_id=job_id, status=JobStatus.FAILED, error="Timeout waiting for result")

    def process_jobs(self, target_types: list[str], handler_func, exit_on_complete: bool = False):
        """(Spoke) Long-polling loop to process jobs matching target_types."""
        print(f"[IPC-Spoke] Listening for jobs: {target_types}")
        while True:
            # Find pending jobs
            for job_file in self.jobs_dir.glob("*.json"):
                if job_file.name.startswith("result_"):
                    continue
                
                try:
                    with open(job_file, "r") as f:
                        data = json.load(f)
                    
                    if data["status"] == JobStatus.PENDING and data["job_type"] in target_types:
                        # Claim job
                        data["status"] = JobStatus.PROCESSING
                        with open(job_file, "w") as f:
                            json.dump(data, f)
                            
                        # Execute
                        print(f"[IPC-Spoke] Processing {data['job_id']}...")
                        try:
                            result_data = handler_func(data["job_type"], data["payload"])
                            result = JobResult(
                                job_id=data["job_id"],
                                status=JobStatus.COMPLETED,
                                result=result_data,
                                completed_at=time.time()
                            )
                        except Exception as e:
                            print(f"[IPC-Spoke] Job failed: {e}")
                            result = JobResult(
                                job_id=data["job_id"],
                                status=JobStatus.FAILED,
                                error=str(e),
                                completed_at=time.time()
                            )
                        
                        # Write result
                        with open(self.jobs_dir / f"result_{data['job_id']}.json", "w") as f:
                            json.dump(asdict(result), f, indent=2)
                            
                        # Cleanup request file
                        job_file.unlink()
                        
                        if exit_on_complete:
                            print("[IPC-Spoke] Job complete. Exiting (One-Shot Mode).")
                            return

                except Exception as e:
                    print(f"[IPC-Spoke] Error in loop: {e}")
                    time.sleep(1)
            
            time.sleep(2)

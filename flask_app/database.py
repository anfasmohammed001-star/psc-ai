"""
File-based database management for processing job persistence.
Maintains history of uploaded files and generation status across server restarts using JobPersister.
"""
import os
import logging
from datetime import datetime, timezone
from glob import glob
from typing import Optional, Dict, Any, List
from sys import platform
from persist.job_persister import JobPersister

logger = logging.getLogger(__name__)

if os.environ.get("VERCEL"):
    JOBS_DIR = "/tmp/jobs_store"
else:
    JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs_store")

def init_db():
    """Initializes the jobs store directory."""
    logger.info("Initializing file-based job database...")
    os.makedirs(JOBS_DIR, exist_ok=True)

def create_job(job_id: str, status: str = "PENDING") -> Dict[str, Any]:
    """Inserts a new job record by writing to a file."""
    job_data = {
        "job_id": job_id,
        "status": status,
        "pages_processed": 0,
        "total_pages": 0,
        "current_stage": "Enqueued",
        "error_message": None,
        "output_pdf_path": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    persister = JobPersister.get_job_persister()
    persister.persist_job(job_data, job_id, JOBS_DIR)
    return job_data

def update_job(job_id: str, **kwargs) -> None:
    """Dynamically updates properties for a specific job_id."""
    job_data = get_job(job_id)
    if not job_data:
        logger.warning(f"Attempted to update non-existent job: {job_id}")
        return
        
    allowed_cols = {"status", "pages_processed", "total_pages", "current_stage", "error_message", "output_pdf_path"}
    
    for col, val in kwargs.items():
        if col in allowed_cols:
            job_data[col] = val
            
    persister = JobPersister.get_job_persister()
    persister.persist_job(job_data, job_id, JOBS_DIR)

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves metadata of a single job."""
    persister = JobPersister.get_job_persister()
    if persister.job_persist_exist(job_id, JOBS_DIR):
        try:
            return persister.load_job(job_id, JOBS_DIR)
        except Exception as e:
            logger.error(f"Error loading job {job_id}: {e}")
            return None
    return None

def get_all_jobs() -> List[Dict[str, Any]]:
    """Retrieves all past jobs sorted by creation time."""
    persister = JobPersister.get_job_persister()
    jobs = []
    
    # We need to find all job files depending on the platform suffix:
    is_on_mac_os = (platform == "darwin")
    suffix = ".mlx.npz" if is_on_mac_os else ".safetensors"
    
    pattern = os.path.join(JOBS_DIR, f"*{suffix}")
    job_files = glob(pattern)
    
    for filepath in job_files:
        filename = os.path.basename(filepath)
        job_id = filename[:-len(suffix)]
        
        # Check if done marker exists to prevent reading incomplete saves
        if persister.job_persist_exist(job_id, JOBS_DIR):
            job_data = get_job(job_id)
            if job_data:
                jobs.append(job_data)
                
    # Sort by created_at in descending order
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return jobs

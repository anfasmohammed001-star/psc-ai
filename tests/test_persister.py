import os
import shutil
import tempfile
import pytest
from sys import platform
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../flask_app')))

from persist.job_persister import JobPersister
from persist.safetensor_job_persister import SafetensorJobPersister
from persist.mlx_job_persister import MlxJobPersister
import database

def test_persister_factory():
    """Verify that JobPersister returns the appropriate class based on the platform."""
    persister = JobPersister.get_job_persister()
    if platform == "darwin":
        assert isinstance(persister, MlxJobPersister)
    else:
        assert isinstance(persister, SafetensorJobPersister)

def test_safetensor_job_persister_lifecycle():
    """Test full create/update/load/exist cycle of SafetensorJobPersister explicitly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        persister = SafetensorJobPersister()
        job_id = "test-job-safetensor-123"
        job_data = {
            "job_id": job_id,
            "status": "PROCESSING",
            "pages_processed": 5,
            "total_pages": 10,
            "current_stage": "Extracting text",
            "error_message": None,
            "output_pdf_path": "/path/to/output.pdf",
            "created_at": "2026-06-04 12:00:00"
        }
        
        # Verify job doesn't exist yet
        assert not persister.job_persist_exist(job_id, temp_dir)
        
        # Persist job
        persister.persist_job(job_data, job_id, temp_dir)
        
        # Check files
        assert os.path.exists(os.path.join(temp_dir, f"{job_id}.safetensors"))
        assert os.path.exists(os.path.join(temp_dir, f"{job_id}.safetensors.done"))
        
        # Verify exist check
        assert persister.job_persist_exist(job_id, temp_dir)
        
        # Load job
        loaded_data = persister.load_job(job_id, temp_dir)
        assert loaded_data == job_data

def test_mlx_job_persister_lifecycle():
    """Test full create/update/load/exist cycle of MlxJobPersister explicitly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        persister = MlxJobPersister()
        job_id = "test-job-mlx-123"
        job_data = {
            "job_id": job_id,
            "status": "PROCESSING",
            "pages_processed": 5,
            "total_pages": 10,
            "current_stage": "Extracting text",
            "error_message": None,
            "output_pdf_path": "/path/to/output.pdf",
            "created_at": "2026-06-04 12:00:00"
        }
        
        # Verify job doesn't exist yet
        assert not persister.job_persist_exist(job_id, temp_dir)
        
        # Persist job
        persister.persist_job(job_data, job_id, temp_dir)
        
        # Check files
        assert os.path.exists(os.path.join(temp_dir, f"{job_id}.mlx.npz"))
        assert os.path.exists(os.path.join(temp_dir, f"{job_id}.mlx.done"))
        
        # Verify exist check
        assert persister.job_persist_exist(job_id, temp_dir)
        
        # Load job
        loaded_data = persister.load_job(job_id, temp_dir)
        assert loaded_data == job_data

def test_done_marker_validation():
    """Verify that exist check returns False if the done marker is missing even if data file exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        persister = SafetensorJobPersister()
        job_id = "incomplete-job"
        job_data = {"job_id": job_id, "status": "PENDING"}
        
        persister.persist_job(job_data, job_id, temp_dir)
        
        # Remove done marker
        done_marker = os.path.join(temp_dir, f"{job_id}.safetensors.done")
        os.remove(done_marker)
        
        # Should now return False because done marker is missing
        assert not persister.job_persist_exist(job_id, temp_dir)

def test_database_module_integration():
    """Verify the database module functions correctly using the JobPersister backend."""
    # Temporarily point database to a temp jobs directory
    with tempfile.TemporaryDirectory() as temp_dir:
        original_jobs_dir = database.JOBS_DIR
        database.JOBS_DIR = temp_dir
        
        try:
            database.init_db()
            
            # Create a job
            job_id = "db-integration-job-1"
            job_data = database.create_job(job_id)
            assert job_data["job_id"] == job_id
            assert job_data["status"] == "PENDING"
            
            # Fetch job
            fetched = database.get_job(job_id)
            assert fetched["job_id"] == job_id
            assert fetched["status"] == "PENDING"
            
            # Update job
            database.update_job(job_id, status="COMPLETED", pages_processed=3, total_pages=3)
            
            # Fetch and check update
            updated = database.get_job(job_id)
            assert updated["status"] == "COMPLETED"
            assert updated["pages_processed"] == 3
            assert updated["total_pages"] == 3
            
            # Get all jobs
            all_jobs = database.get_all_jobs()
            assert len(all_jobs) == 1
            assert all_jobs[0]["job_id"] == job_id
            
        finally:
            database.JOBS_DIR = original_jobs_dir

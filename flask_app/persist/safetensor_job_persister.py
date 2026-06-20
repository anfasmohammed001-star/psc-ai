import os
import json
from pathlib import Path
from .job_persister import JobPersister

class SafetensorJobPersister(JobPersister):
    def __init__(self, *args, **kwargs):
        super(SafetensorJobPersister, self).__init__(*args, **kwargs)

    def job_persist_exist(self, job_id, saving_path):
        saving_path = Path(saving_path)
        safetensor_exists = os.path.exists(str(saving_path / (job_id + '.safetensors')))
        done_marker_exists = os.path.exists(str(saving_path / (job_id + '.safetensors.done')))
        return safetensor_exists and done_marker_exists

    def persist_job(self, job_data, job_id, saving_path):
        saving_path = Path(saving_path)
        saving_path.mkdir(parents=True, exist_ok=True)
        file_path = saving_path / (job_id + '.safetensors')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(job_data, f, ensure_ascii=False, indent=2)
            
        print(f"saved as: {file_path}")
        
        # Set done marker
        (saving_path / (job_id + '.safetensors.done')).touch()

    def load_job(self, job_id, path):
        path = Path(path)
        file_path = path / (job_id + '.safetensors')
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

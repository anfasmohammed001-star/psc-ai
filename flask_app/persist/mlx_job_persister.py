import os
from pathlib import Path
import numpy as np
from .job_persister import JobPersister

class MlxJobPersister(JobPersister):
    def __init__(self, *args, **kwargs):
        super(MlxJobPersister, self).__init__(*args, **kwargs)

    def job_persist_exist(self, job_id, saving_path):
        saving_path = Path(saving_path)
        npz_exists = os.path.exists(str(saving_path / (job_id + '.mlx.npz')))
        done_marker_exists = os.path.exists(str(saving_path / (job_id + '.mlx.done')))
        return npz_exists and done_marker_exists

    def persist_job(self, job_data, job_id, saving_path):
        saving_path = Path(saving_path)
        saving_path.mkdir(parents=True, exist_ok=True)
        file_path = saving_path / (job_id + '.mlx')
        
        # Prepare state dict of numpy arrays
        np_dict = {}
        for k, v in job_data.items():
            if v is None:
                np_dict[k] = np.array("__NONE__", dtype=object)
            else:
                np_dict[k] = np.array(v)
                
        np.savez(file_path, **np_dict)
        print(f"saved as: {file_path}.npz")
        
        # Set done marker
        (saving_path / (job_id + '.mlx.done')).touch()

    def load_job(self, job_id, path):
        path = Path(path)
        file_path = path / (job_id + '.mlx.npz')
        
        with np.load(file_path, allow_pickle=True) as data:
            job_data = {}
            for k in data.files:
                val = data[k]
                item = val.item() if val.ndim == 0 else val.tolist()
                
                # Check for special None marker
                if item == "__NONE__":
                    item = None
                elif isinstance(item, bytes):
                    item = item.decode('utf-8')
                
                # Ensure correct Python types for fields (convert NumPy types)
                if isinstance(item, (np.integer, np.int64, np.int32)):
                    item = int(item)
                elif isinstance(item, (np.floating, np.float64, np.float32)):
                    item = float(item)
                
                job_data[k] = item
                
            return job_data

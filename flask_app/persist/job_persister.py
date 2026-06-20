import os
from sys import platform

job_persister = None

class JobPersister:
    def __init__(self):
        pass

    @classmethod
    def get_job_persister(cls):
        global job_persister
        if job_persister is not None:
            return job_persister

        is_on_mac_os = False
        if platform == "darwin":
            is_on_mac_os = True

        if is_on_mac_os:
            from .mlx_job_persister import MlxJobPersister
            job_persister = MlxJobPersister()
        else:
            from .safetensor_job_persister import SafetensorJobPersister
            job_persister = SafetensorJobPersister()
        return job_persister

    def job_persist_exist(self, job_id, saving_path):
        pass

    def persist_job(self, job_data, job_id, path):
        pass

    def load_job(self, job_id, path):
        pass

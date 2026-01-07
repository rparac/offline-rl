import os
from typing import Callable
import numpy as np

import wandb
import gymnasium as gym
from gymnasium import error, logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

class WandbAutoUploadCallback(BaseCallback):
    def __init__(self, video_folder, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.check_freq = check_freq
        self.uploaded_files = set()  # Keep track of what we've sent

    def _on_step(self) -> bool:
        # Only check every N steps to avoid slowing down training with disk I/O
        if self.n_calls % self.check_freq == 0:
            self._scan_and_upload()
        return True

    def _on_training_end(self) -> None:
        # Final check to catch the last video
        self._scan_and_upload()

    def _scan_and_upload(self):
        if not os.path.exists(self.video_folder):
            return

        # List all mp4 files
        files = [f for f in os.listdir(self.video_folder) if f.endswith(".mp4")]
        
        for file in files:
            file_path = os.path.join(self.video_folder, file)
            
            # If we haven't uploaded this file yet...
            if file not in self.uploaded_files:
                # OPTIONAL: Check if file is fully written (simple heuristic: size > 0)
                if os.path.getsize(file_path) > 0:
                    if self.verbose > 0:
                        print(f"Uploading new video: {file}")
                    
                    wandb.log({"video": wandb.Video(file_path, format="mp4")})
                    self.uploaded_files.add(file)

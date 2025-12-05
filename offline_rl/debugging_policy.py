
import time

import torch
from offline_rl.dataset_generator import QTablePolicy
from pathlib import Path
import numpy as np
from PIL import Image

from offline_rl.env_util import setup_visual_minecraft

q_table_path = Path("artifacts/q_table.npy")
q_table = np.load(q_table_path)
print(f"Loaded Q-table with shape: {q_table.shape}")
q_table_policy_module = QTablePolicy(q_table)


env = setup_visual_minecraft()

seed = 2

obs, info = env.reset(seed=seed)

done = False
while not done:
    obs = torch.tensor(obs)
    action = q_table_policy_module(obs.unsqueeze(0))
    act = action.item()
    obs, reward, terminated, truncated, info = env.step(act)

    img = env.render()
    if img is not None:
        im = Image.fromarray(img)
        im.save(f"debug_step_{int(time.time()*1000)}.png")
    done = terminated or truncated


    print(obs)
    print(reward)
    print(terminated)
    print(truncated)
    print(info)
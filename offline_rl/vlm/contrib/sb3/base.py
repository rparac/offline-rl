from typing import Type

from offline_rl.vlm.contrib.sb3.clip_rewarded_dqn import CLIPRewardedDQN
from offline_rl.vlm.contrib.sb3.clip_rewarded_sac import CLIPRewardedSAC


def get_clip_rewarded_rl_algorithm_class(env_name: str) -> Type[CLIPRewardedSAC]:
    """Get the CLIP-rewarded RL algorithm class for a given environment.
    
    Args:
        env_name: The name of the environment
        
    Returns:
        The CLIP-rewarded RL algorithm class (currently always CLIPRewardedSAC)
    """
    if env_name == "VisualMinecraft-v0":
        return CLIPRewardedDQN
    return CLIPRewardedSAC

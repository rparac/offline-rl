import os
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# Set environment variables (from wrapper script)
os.environ.setdefault('DISPLAY', ':50')
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'kms_swrast'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['DISABLE_VULKAN'] = '1'

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
while True:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    random_actions = [np.random.choice(admissible_commands[0])]

    # step
    obs, scores, dones, infos = env.step(random_actions)
    print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
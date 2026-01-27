# ALFWorld Task Specification Guide

## 🎯 How Tasks Work in ALFWorld

ALFWorld loads tasks from JSON trajectory files stored in the dataset directory. By default, tasks are **randomly selected** from the available tasks that match your configuration.

## 📋 Task Types

ALFWorld has 6 task types (defined in `base_config.yaml`):

| ID | Task Type Name | Description |
|---|----------------|-------------|
| 1 | Pick & Place | Pick up an object and place it in a receptacle |
| 2 | Examine in Light | Find an object and look at it under light |
| 3 | Clean & Place | Clean an object then place it in a receptacle |
| 4 | Heat & Place | Heat an object then place it in a receptacle |
| 5 | Cool & Place | Cool an object then place it in a receptacle |
| 6 | Pick Two & Place | Pick up two objects and place them in a receptacle together |

## 🔧 Filtering Task Types (Config-Based)

You can filter which **task types** to use by modifying `task_types` in your config file:

### Method 1: Modify `base_config.yaml`

```yaml
env:
  task_types: [1]  # Only Pick & Place tasks
  # task_types: [3, 4]  # Only Clean & Place and Heat & Place
  # task_types: [1, 2, 3, 4, 5, 6]  # All task types (default)
```

### Method 2: Modify Config in Code

```python
import alfworld.agents.modules.generic as generic

# Load config
config = generic.load_config()

# Filter to only specific task types
config['env']['task_types'] = [1]  # Only Pick & Place

# Or filter to multiple types
config['env']['task_types'] = [3, 4]  # Clean & Place and Heat & Place

# Now use this config
from alfworld.agents.environment import get_environment
env = get_environment(config['env']['type'])(config, train_eval='train')
env = env.init_env(batch_size=1)
```

**Note**: This still randomly selects from the filtered task types. To get the same task each time, you need to control the random seed (see below).

## 🎲 Controlling Randomness (Same Task Each Time)

To get **reproducible tasks**, use a fixed random seed:

### Method 1: Set Seed in Config

```yaml
general:
  random_seed: 42  # Fixed seed for reproducibility
```

### Method 2: Set Seed in Code

```python
import numpy as np
import alfworld.agents.modules.generic as generic

# Set seed before loading config
np.random.seed(42)

# Load config
config = generic.load_config()
config['general']['random_seed'] = 42

# Setup environment
from alfworld.agents.environment import get_environment
env = get_environment(config['env']['type'])(config, train_eval='train')
env = env.init_env(batch_size=1)

# Reset with same seed
obs, info = env.reset()
```

## 📁 Task JSON Files Location

Tasks are stored as JSON files in the ALFWorld cache directory:

```
~/.cache/alfworld/json_2.1.1/
├── train/              # Training tasks
├── valid_seen/         # Validation (seen scenes)
└── valid_unseen/       # Validation (unseen scenes)
```

Each task has a structure like:
```
pick_and_place_simple-AppleSliced-None-Fridge-30/
└── trial_T20190907_105723_672392/
    └── traj_data.json
```

## 🔍 Finding Specific Tasks

To find tasks of a specific type or with specific objects:

```python
import os
import json
import glob

# Find all JSON files
alfworld_data = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
json_files = glob.glob(f"{alfworld_data}/json_2.1.1/train/**/traj_data.json", recursive=True)

# Load and inspect tasks
for json_file in json_files[:5]:  # Check first 5
    with open(json_file, 'r') as f:
        traj = json.load(f)
    print(f"Task: {traj['task_type']}, Scene: {traj['scene']['scene_num']}")
    print(f"Goal: {traj['turk_annotations']['anns'][0]['task_desc']}")
    print(f"File: {json_file}\n")
```

## 💡 Current Configuration

Looking at your `base_config.yaml`:

```yaml
env:
  task_types: [1, 2, 3, 4, 5, 6]  # All task types enabled
  
general:
  random_seed: 42  # Fixed seed (good for reproducibility!)
```

This means:
- ✅ All task types are enabled
- ✅ Tasks are randomly selected from all 6 types
- ✅ With seed 42, the sequence should be reproducible
- ❌ But you can't specify a specific JSON file directly

## 🎯 Recommendations

### For Reproducible Experiments:
1. Set a fixed `random_seed` in config (you already have this!)
2. Tasks will be selected in the same order when you run multiple times

### For Specific Task Type Testing:
1. Set `task_types: [1]` to only get Pick & Place tasks
2. Use fixed seed for reproducibility

### For Specific Task File (Advanced):
If you need a specific JSON file, you'd need to:
1. Modify ALFWorld's environment wrapper code to accept a specific JSON path
2. Or manually filter the dataset files before loading

## 📝 Example: Restrict to One Task Type

```python
import os
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# Set seed for reproducibility
np.random.seed(42)

# Load config
config = generic.load_config()

# Restrict to only "Pick & Place" tasks (type 1)
config['env']['task_types'] = [1]

# Setup environment
env_type = config['env']['type']
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# Reset - will randomly select from Pick & Place tasks only
obs, info = env.reset()
print(f"Task description: {obs[0]}")

# With same seed, you'll get the same sequence of tasks
```

## ⚠️ Important Notes

1. **Random Selection**: Even with filtered task types, tasks are randomly selected from matching files
2. **Seed Matters**: Use the same seed to get reproducible task sequences
3. **Task Types vs Specific Tasks**: `task_types` filters categories, not individual tasks
4. **JSON Structure**: Each `traj_data.json` contains full trajectory data including scene, objects, and goal

## 🚀 Summary

- **Filter task types**: Modify `config['env']['task_types']` (e.g., `[1]` for only Pick & Place)
- **Reproducibility**: Use fixed `random_seed` in config
- **Random selection**: Tasks are randomly chosen from filtered types
- **Specific tasks**: Not directly supported; use seed + task type filtering

Enjoy exploring ALFWorld! 🏠

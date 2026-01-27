# ALFWorld Image-Based Observations Guide

## ✅ Yes, ALFWorld Supports Image Observations!

ALFWorld **does support image-based observations** when using the **THOR (AlfredThorEnv)** or **Hybrid (AlfredHybrid)** environment types.

## 📋 Environment Types & Observation Modes

| Environment Type | Observation Mode | Description |
|------------------|------------------|-------------|
| **AlfredTWEnv** | Text-only | Symbolic descriptions, no images |
| **AlfredThorEnv** | **Visual (RGB images)** | Full visual observations from AI2-THOR |
| **AlfredHybrid** | Mixed (text + visual) | Switches between text and visual modalities |

## 🖼️ How to Access Images

### Basic Access

ALFWorld's THOR environment extends AI2Thor's `Controller`. After each `step()`, the returned `event` object contains the image frame:

```python
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# Load config
config = generic.load_config()
env_type = config['env']['type']  # Must be 'AlfredThorEnv' or 'AlfredHybrid'

# Setup environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# Reset and get initial observation
obs, info = env.reset()

# Step through environment
action = "go to countertop 1"
obs, scores, dones, infos = env.step([action])

# Access image from the last event
# The environment's internal ThorEnv controller stores the last event
if hasattr(env, 'env') and hasattr(env.env, 'last_event'):
    frame = env.env.last_event.frame
    # Frame is numpy array: shape (height, width, 3) in BGR format
    print(f"Image shape: {frame.shape}")  # e.g., (300, 300, 3)
    print(f"Image dtype: {frame.dtype}")  # e.g., uint8
```

### Converting BGR to RGB

The frames from AI2Thor are in **BGR format** (not RGB). To convert to RGB:

```python
import numpy as np
import cv2

# Option 1: Reverse channels manually
rgb_frame = frame[:, :, ::-1]  # BGR -> RGB

# Option 2: Use cv2 (if available)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### Image Properties

- **Format**: BGR (Blue-Green-Red) - standard for OpenCV
- **Shape**: `(height, width, 3)` - typically (300, 300, 3) based on config
- **Dtype**: `uint8` - values from 0-255
- **Access**: `event.frame` or `env.env.last_event.frame`

### Example: Visualizing Observations

```python
import matplotlib.pyplot as plt
import numpy as np

# After a step
obs, scores, dones, infos = env.step([action])

# Get the image frame (BGR format)
frame_bgr = env.env.last_event.frame

# Convert to RGB for display
frame_rgb = frame_bgr[:, :, ::-1]

# Display
plt.figure(figsize=(8, 8))
plt.imshow(frame_rgb)
plt.axis('off')
plt.title('ALFWorld Observation')
plt.show()

# Or save to file
import cv2
cv2.imwrite('observation.png', frame_rgb)
```

## 🔧 Configuration

### Enabling Image Rendering

In your config (`base_config.yaml`):

```yaml
env:
  type: 'AlfredThorEnv'  # Must be THOR or Hybrid for images
  
  thor:
    screen_width: 300     # Image width
    screen_height: 300    # Image height
    save_frames_to_disk: False  # Set True to save frames
    save_frames_path: './videos/'  # Where to save frames
```

### Vision Model Configuration

ALFWorld also supports vision-based agents that use image features:

```yaml
vision_dagger:
  model_type: "resnet"  # Options: "resnet", "maskrcnn_whole", "maskrcnn", "no_vision"
  resnet_fc_dim: 64
  maskrcnn_top_k_boxes: 10
```

## 🎯 Complete Example

```python
import os
import numpy as np

# Set environment variables (from wrapper script)
os.environ.setdefault('DISPLAY', ':50')
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
# ... other env vars ...

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# Load config - ensure env type is AlfredThorEnv
config = generic.load_config()
print(f"Environment type: {config['env']['type']}")

# Setup environment
env = get_environment(config['env']['type'])(config, train_eval='train')
env = env.init_env(batch_size=1)

# Reset
obs, info = env.reset()
print(f"Initial obs type: {type(obs)}")
print(f"Initial obs: {obs[0][:100]}")  # First 100 chars (text obs)

# Access image frame
if hasattr(env, 'env') and hasattr(env.env, 'last_event'):
    frame = env.env.last_event.frame
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame range: [{frame.min()}, {frame.max()}]")

# Take a step
action = "go to countertop 1"
obs, scores, dones, infos = env.step([action])

# Access updated frame
if hasattr(env, 'env') and hasattr(env.env, 'last_event'):
    new_frame = env.env.last_event.frame
    print(f"New frame shape: {new_frame.shape}")
```

## 📝 Important Notes

1. **Observation Format**: The `obs` returned by `env.step()` is **text-based** (symbolic description), not images. To get images, access `env.env.last_event.frame` directly.

2. **Text vs Image Observations**:
   - `obs` = Text description (e.g., "You are in a kitchen. On the countertop, you see...")
   - `env.env.last_event.frame` = RGB image array (numpy array)

3. **Hybrid Mode**: In `AlfredHybrid`, the environment switches between text and visual modes probabilistically.

4. **Agent Types**: 
   - Text-based agents use `obs` (text descriptions)
   - Vision-based agents (e.g., `VisionDaggerAgent`) use image frames directly

5. **Performance**: Visual rendering is slower than text-only mode, but provides richer observations.

## 🚀 Next Steps

- Check `base_config.yaml` to ensure `env.type` is `'AlfredThorEnv'`
- Access frames via `env.env.last_event.frame` after each step
- Convert BGR to RGB if needed: `frame[:, :, ::-1]`
- Use vision-based agents if you want to train on images directly

Enjoy exploring ALFWorld with visual observations! 🖼️

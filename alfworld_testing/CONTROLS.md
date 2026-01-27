# ALFWorld / AI2-THOR Controls Quick Reference

## 🎮 Text Commands (Terminal Interface)

**Note**: `alfworld-play-thor` uses **text commands in the terminal**, not keyboard controls. The Unity window is for visualization only.

### Basic Movement & Navigation
- **`go to <location> <index>`**: Move agent to a location
  - Example: `go to countertop 1`, `go to drawer 2`
- **`look`** or **`look around`**: Get description of current view
- **`inventory`**: Show what the agent is currently carrying

### Object Interactions
- **`take <object> <index> from <location>`**: Pick up an object
  - Example: `take apple 1 from countertop 2`
- **`put <object> <index> in/on <location>`**: Place an object
  - Example: `put apple 1 in sinkbasin 1`
- **`open <object> <index>`**: Open a receptacle (drawer, cabinet, etc.)
- **`close <object> <index>`**: Close a receptacle

### Actions
- **`heat <object> <index> with <appliance>`**: Heat an object
  - Example: `heat potato 1 with microwave 1`
- **`cool <object> <index> with <appliance>`**: Cool an object
- **`clean <object> <index> with <appliance>`**: Clean an object
  - Example: `clean plate 1 with sinkbasin 1`
- **`examine <object> <index>`**: Get detailed info about an object

### Utility Commands
- **`help`**: Show available commands
- **`quit`** or **`exit`**: End the session
- **`Ctrl+C`**: Force quit

## 💡 Tips

1. **Type commands in the terminal**: The terminal where you ran `alfworld-play-thor` accepts text commands. The Unity window is just for visualization.

2. **Use `help` command**: Type `help` in the terminal to see all available commands for the current state.

3. **Object indices**: Objects are numbered (e.g., `apple 1`, `apple 2`). Use `look` or `inventory` to see what's available.

4. **Watch the Unity window**: As you type commands, you'll see the agent move and interact in the Unity visualization window.

5. **Programmatic Control**: If you want to control via Python code instead of text commands:
   ```python
   from alfworld.agents.environment import get_environment
   
   env = get_environment("AlfredThorEnv")(config, train_eval='train')
   env = env.init_env(batch_size=1)
   obs, info = env.reset()
   
   # Execute actions programmatically
   obs, scores, dones, infos = env.step(["go to countertop 1"])
   ```

## 🎯 Example Workflow

1. **Start the game**: `./alfworld_testing/alfworld_wrapper.sh`
2. **Read the task**: The terminal will show your task (e.g., "put a hot potato in sinkbasin")
3. **Type commands**: 
   - `go to countertop 1`
   - `take potato 1 from countertop 1`
   - `go to microwave 1`
   - `heat potato 1 with microwave 1`
   - `go to sinkbasin 1`
   - `put potato 1 in sinkbasin 1`
4. **Use `help`**: If stuck, type `help` to see valid commands
5. **Quit**: Type `quit` or press `Ctrl+C`

## 📝 Example Commands

- `go to drawer 1` - Navigate to a location
- `take apple 1 from countertop 2` - Pick up an object
- `put apple 1 in sinkbasin 1` - Place an object
- `open drawer 1` - Open a receptacle
- `close drawer 1` - Close a receptacle
- `heat potato 1 with microwave 1` - Heat an object
- `clean plate 1 with sinkbasin 1` - Clean an object
- `inventory` - Show what you're carrying
- `look` - Describe current view

Enjoy exploring ALFWorld! 🏠

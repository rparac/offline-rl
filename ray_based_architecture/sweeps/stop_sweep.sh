#!/bin/bash
# Stop all running sweep agents and their training processes

echo "Stopping all wandb sweep agents and training processes..."
echo ""

# Find all wandb agent processes
AGENT_PIDS=$(pgrep -f "wandb agent")

# Find all Python training processes (discrete_sac.py)
TRAINING_PIDS=$(pgrep -f "discrete_sac.py")

# Find all Ray processes spawned by these runs
RAY_PIDS=$(pgrep -f "ray::")

if [ -z "$AGENT_PIDS" ] && [ -z "$TRAINING_PIDS" ] && [ -z "$RAY_PIDS" ]; then
    echo "No running sweep agents or training processes found"
    exit 0
fi

# Display what we found
if [ -n "$AGENT_PIDS" ]; then
    echo "Found wandb agent PIDs: $AGENT_PIDS"
fi
if [ -n "$TRAINING_PIDS" ]; then
    echo "Found training script PIDs: $TRAINING_PIDS"
fi
if [ -n "$RAY_PIDS" ]; then
    RAY_COUNT=$(echo "$RAY_PIDS" | wc -w)
    echo "Found $RAY_COUNT Ray processes"
fi
echo ""

# Step 1: Kill wandb agents gracefully (they should clean up their children)
if [ -n "$AGENT_PIDS" ]; then
    echo "Sending SIGTERM to wandb agents..."
    pkill -TERM -f "wandb agent"
    sleep 2
fi

# Step 2: Kill training processes gracefully (triggers finally block cleanup)
if [ -n "$TRAINING_PIDS" ]; then
    echo "Sending SIGTERM to training processes..."
    pkill -TERM -f "discrete_sac.py"
    sleep 3
fi

# Step 3: Check if any are still running and force kill
REMAINING_AGENTS=$(pgrep -f "wandb agent")
REMAINING_TRAINING=$(pgrep -f "discrete_sac.py")

if [ -n "$REMAINING_AGENTS" ]; then
    echo "Some agents still running, sending SIGKILL..."
    pkill -KILL -f "wandb agent"
    sleep 1
fi

if [ -n "$REMAINING_TRAINING" ]; then
    echo "Some training processes still running, sending SIGKILL..."
    pkill -KILL -f "discrete_sac.py"
    sleep 1
fi

# Step 4: Clean up any orphaned Ray processes
REMAINING_RAY=$(pgrep -f "ray::")
if [ -n "$REMAINING_RAY" ]; then
    echo "Cleaning up orphaned Ray processes..."
    pkill -TERM -f "ray::"
    sleep 2
    # Force kill if still there
    STILL_RAY=$(pgrep -f "ray::")
    if [ -n "$STILL_RAY" ]; then
        pkill -KILL -f "ray::"
    fi
fi

# Final verification
echo ""
echo "Verifying cleanup..."
STILL_AGENTS=$(pgrep -f "wandb agent")
STILL_TRAINING=$(pgrep -f "discrete_sac.py")
STILL_RAY=$(pgrep -f "ray::")

if [ -z "$STILL_AGENTS" ] && [ -z "$STILL_TRAINING" ] && [ -z "$STILL_RAY" ]; then
    echo "✓ All sweep agents and training processes stopped successfully"
else
    echo "✗ Warning: Some processes may still be running:"
    [ -n "$STILL_AGENTS" ] && echo "  - Agents: $STILL_AGENTS"
    [ -n "$STILL_TRAINING" ] && echo "  - Training: $STILL_TRAINING"
    [ -n "$STILL_RAY" ] && echo "  - Ray processes: $(echo $STILL_RAY | wc -w) remaining"
    echo ""
    echo "Run 'ps aux | grep -E \"wandb agent|discrete_sac|ray::\"' for details"
    exit 1
fi

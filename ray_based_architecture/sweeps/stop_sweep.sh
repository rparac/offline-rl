#!/bin/bash
# Stop all running sweep agents

echo "Stopping all wandb sweep agents..."

# Find all wandb agent processes
AGENT_PIDS=$(pgrep -f "wandb agent")

if [ -z "$AGENT_PIDS" ]; then
    echo "No running sweep agents found"
    exit 0
fi

echo "Found agents with PIDs: $AGENT_PIDS"
echo ""

# Kill agents gracefully first (SIGTERM)
echo "Sending SIGTERM to agents..."
pkill -TERM -f "wandb agent"

# Wait a bit for graceful shutdown
sleep 3

# Check if any are still running
REMAINING=$(pgrep -f "wandb agent")

if [ -n "$REMAINING" ]; then
    echo "Some agents still running, sending SIGKILL..."
    pkill -KILL -f "wandb agent"
    sleep 1
fi

# Verify all stopped
STILL_RUNNING=$(pgrep -f "wandb agent")

if [ -z "$STILL_RUNNING" ]; then
    echo "✓ All agents stopped successfully"
else
    echo "✗ Warning: Some agents may still be running (PIDs: $STILL_RUNNING)"
    exit 1
fi

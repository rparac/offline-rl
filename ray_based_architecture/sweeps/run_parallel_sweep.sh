#!/bin/bash
# Run multiple wandb sweep agents in parallel, each on a different GPU
#
# Usage:
#   1. Create sweep: wandb sweep ray_based_architecture/sweeps/sac_atari_hparams.yaml
#   2. Run this script: bash ray_based_architecture/sweeps/run_parallel_sweep.sh <sweep_id> [num_agents]
#
# Example:
#   bash ray_based_architecture/sweeps/run_parallel_sweep.sh username/project/abc123 4

if [ -z "$1" ]; then
    echo "Error: Sweep ID required"
    echo "Usage: $0 <sweep_id> [num_agents]"
    echo ""
    echo "First create a sweep:"
    echo "  wandb sweep ray_based_architecture/sweeps/sac_atari_hparams.yaml"
    echo ""
    echo "Then run this script with the sweep ID:"
    echo "  $0 username/project/sweep_id 4"
    exit 1
fi

SWEEP_ID=$1
NUM_AGENTS=${2:-4}  # Default to 4 agents

echo "Starting $NUM_AGENTS parallel sweep agents for sweep: $SWEEP_ID"
echo "Each agent will use 1 GPU"
echo ""

# Create logs directory
mkdir -p logs/sweep_agents

# Launch agents in background
for i in $(seq 0 $((NUM_AGENTS-1))); do
    GPU_ID=$i
    LOG_FILE="logs/sweep_agents/agent_gpu${GPU_ID}.log"
    
    echo "Launching agent $i on GPU $GPU_ID (log: $LOG_FILE)"
    
    # Run agent in background, redirecting output to log file
    CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent $SWEEP_ID > $LOG_FILE 2>&1 &
    
    AGENT_PID=$!
    echo "  → Agent PID: $AGENT_PID"
    
    # Small delay to avoid startup race conditions
    sleep 2
done

echo ""
echo "All agents launched!"
echo ""
echo "Commands:"
echo "  Check status:      bash ray_based_architecture/sweeps/check_sweep_status.sh"
echo "  View logs:         tail -f logs/sweep_agents/agent_gpu*.log"
echo "  Stop all agents:   bash ray_based_architecture/sweeps/stop_sweep.sh"

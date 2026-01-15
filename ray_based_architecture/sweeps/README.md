# Parallel Hyperparameter Sweeps with WandB

This directory contains tools for running parallel hyperparameter sweeps across multiple GPUs.

## Quick Start (4 GPUs in parallel)

```bash
# 1. Create the sweep
wandb sweep ray_based_architecture/sweeps/sac_atari_hparams.yaml

# 2. Launch 4 parallel agents (copy the sweep ID from step 1)
bash ray_based_architecture/sweeps/run_parallel_sweep.sh <sweep_id> 4

# 3. Monitor progress
bash ray_based_architecture/sweeps/monitor_sweep.sh

# 4. Stop all agents when done
bash ray_based_architecture/sweeps/stop_sweep.sh
```

## Detailed Instructions

### 1. Create a Sweep

```bash
cd /home/rp218/projects/offline-rl
wandb sweep ray_based_architecture/sweeps/sac_atari_hparams.yaml
```

This creates a sweep and outputs a sweep ID like: `username/project/abc123xyz`

### 2. Launch Parallel Agents

**Option A: Use the helper script (recommended)**
```bash
bash ray_based_architecture/sweeps/run_parallel_sweep.sh <sweep_id> 4
```

**Option B: Manual launch in separate terminals**
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>

# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id>

# Terminal 3
CUDA_VISIBLE_DEVICES=2 wandb agent <sweep_id>

# Terminal 4
CUDA_VISIBLE_DEVICES=3 wandb agent <sweep_id>
```

### 3. Monitor Progress

**Check status of all processes:**
```bash
bash ray_based_architecture/sweeps/check_sweep_status.sh
```

**View logs in real-time:**
```bash
tail -f logs/sweep_agents/agent_gpu*.log
```

**Watch continuously (updates every 5 seconds):**
```bash
watch -n 5 bash ray_based_architecture/sweeps/check_sweep_status.sh
```

**View WandB dashboard:**
```bash
# Visit your sweep page on wandb.ai
```

### 4. Stop Agents

**Stop all agents and training processes:**
```bash
bash ray_based_architecture/sweeps/stop_sweep.sh
```

This will:
1. Send SIGTERM to wandb agents (graceful shutdown)
2. Send SIGTERM to Python training scripts (triggers cleanup)
3. Force kill any remaining processes with SIGKILL
4. Clean up orphaned Ray processes

**Check if anything is still running:**
```bash
bash ray_based_architecture/sweeps/check_sweep_status.sh
```

## How It Works

1. **GPU Isolation**: Each agent runs with `CUDA_VISIBLE_DEVICES` set to a specific GPU, ensuring no GPU contention
2. **Independent Ray Instances**: Each training run starts its own Ray instance (controlled by `--use-multiple-agents` flag)
3. **Shared Sweep Queue**: All agents pull hyperparameter configs from the same WandB sweep
4. **Parallel Execution**: Each agent trains models independently in parallel

### Ray Initialization Modes

The `--use-multiple-agents` flag controls Ray initialization:

- **`--use-multiple-agents`** (parallel sweeps): Each training run calls `ray.init()` and `ray.shutdown()`
  - ✅ Use for: WandB parallel sweeps, isolated experiments
  - ✅ Each agent gets independent Ray instance
  - ✅ Clean resource cleanup after each run

- **No flag** (single run): Uses existing Ray cluster without init/shutdown
  - ✅ Use for: Single training runs, debugging with persistent Ray cluster
  - ✅ Faster iteration (no Ray startup overhead)
  - ✅ Can inspect Ray dashboard between runs

## Resource Requirements

- **Per agent**: 1 GPU
- **Ray overhead**: ~2-4 CPU cores per agent
- **Memory**: Depends on model size and replay buffer (typically 8-16 GB per agent)

With 4 GPUs, you can run ~4x faster sweeps!

## Customization

### Modify sweep parameters

Edit `sac_atari_hparams.yaml`:
```yaml
parameters:
  batch-size:
    values: [256, 512, 1024]  # Add/remove values
  policy-lr:
    min: 1e-5
    max: 3e-4
```

### Run with fewer/more agents

```bash
# Run with 2 agents (use GPUs 0 and 1)
bash ray_based_architecture/sweeps/run_parallel_sweep.sh <sweep_id> 2

# Run with 8 agents (if you have 8 GPUs)
bash ray_based_architecture/sweeps/run_parallel_sweep.sh <sweep_id> 8
```

## Troubleshooting

### Agents not starting
- Check GPU availability: `nvidia-smi`
- Check Ray status: `ray status`
- View logs: `tail -f logs/sweep_agents/agent_gpu0.log`

### Out of memory errors
- Reduce `--num-envs` in `sac_atari_hparams.yaml`
- Reduce `--buffer-size`
- Reduce `--batch-size` range

### Ray connection issues

**For parallel sweeps**, the `--use-multiple-agents` flag is already set in `sac_atari_hparams.yaml`.

**For single runs**, connect to Ray first:
```bash
ray start --head  # Or ray.init(address='auto') in code
python ray_based_architecture/rl/discrete_sac.py  # Without --use-multiple-agents
```

### Cleanup stale Ray processes
```bash
ray stop  # Stop Ray cluster
pkill -f ray::  # Kill stale Ray processes
```

## Best Practices

1. **Start small**: Test with 1-2 agents first to verify everything works
2. **Monitor resources**: Use `htop` and `nvidia-smi` to ensure you're not overloading the system
3. **Log everything**: All agent output goes to `logs/sweep_agents/`, check these if things fail
4. **Set limits**: In `sac_atari_hparams.yaml`, set `--total-timesteps` appropriately (shorter for quick tests)
5. **Use tmux/screen**: For long-running sweeps, use tmux so agents survive terminal disconnects

## Example: Full Workflow

```bash
# 1. Create sweep
wandb sweep ray_based_architecture/sweeps/sac_atari_hparams.yaml
# Output: Created sweep: username/project/abc123

# 2. Start 4 agents
bash ray_based_architecture/sweeps/run_parallel_sweep.sh username/project/abc123 4

# 3. Monitor in another terminal
watch -n 10 bash ray_based_architecture/sweeps/monitor_sweep.sh

# 4. View logs
tail -f logs/sweep_agents/agent_gpu*.log

# 5. When done (or to stop early)
bash ray_based_architecture/sweeps/stop_sweep.sh
```

Enjoy your 4x speedup! 🚀

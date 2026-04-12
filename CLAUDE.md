# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning (MARL) project implementing **MAPPO (Multi-Agent Proximal Policy Optimization)** with **Centralized Training with Decentralized Execution (CTDE)** for swarm formation control. Agents learn to form geometric shapes (circle, square, triangle, line, etc.) on a 64×64 grid. Target positions are either LLM-generated or built-in.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"  # only needed for LLM shape generation

# Train (defaults: CNN actor, 4 agents, circle, 1000 episodes)
python main.py --mode train --shape circle --n_agents 8 --n_episodes 1000

# Train with improved MAPPO (value normalization, Huber loss)
python main.py --mode train --trainer improved --shape circle --n_agents 8 --n_episodes 5000

# Train with random targets for generalization (preferred for general policy)
python main.py --mode train --trainer improved --random_targets --n_agents 8 --n_episodes 10000 --actor_type mlp

# Resume training from a checkpoint (adds n_episodes on top of resume_episode)
python main.py --mode train --trainer improved --random_targets --n_agents 8 --n_episodes 8300 --actor_type mlp --resume_episode 1700

# Multi-shape training (intermediate between fixed shape and random)
python main.py --mode train --trainer improved --train_shapes circle,triangle,square,line --n_agents 8 --n_episodes 5000

# Evaluate (--obs_radius and --actor_type must match the training run)
python main.py --mode eval --shape circle --n_agents 8 --visualize
python main.py --mode eval --trainer improved --random_targets --n_agents 8 --actor_type mlp --visualize

# Preview target coordinates before training
python shape_preview.py --shape circle --n_agents 8

# Run CBS classical planner baseline
python cbs_solver.py --shape triangle --n_agents 8 --vis_dir visualizations/cbs

# LED matrix visualizer (Raspberry Pi)
python pi/interactive_display.py --text-input                          # MAPPO policy, text input
python pi/interactive_display.py --text-input --cbs --llm-agents       # CBS planner, LLM chooses agent count
```

Key flags: `--mode` (train/eval/demo), `--trainer` (baseline/improved), `--actor_type` (cnn/mlp), `--n_agents`, `--shape`, `--n_episodes`, `--obs_radius` (default 5 → 11×11 local obs), `--device` (auto/cuda/cpu), `--visualize`, `--vis_dir`, `--resume_episode` (improved trainer only).

## Architecture

### Training Flow
1. **Target generation** — `llm/shape_gen.py` calls OpenAI API or uses built-in geometry to produce `[x, y]` coordinates for each agent's target
2. **Environment** — `environment/grid_env.py` implements a PettingZoo parallel env on a 64×64 grid with 9 discrete actions (stay + 8 directions)
3. **MAPPO loop** — rollout → GAE computation (centralized critic, global state) → PPO update (10 epochs, clipped surrogate + entropy)
4. **Evaluation** — deterministic policy, collision tracking, optional GIF/plot output via `environment/visualize.py`

### Key Files
| File | Role |
|------|------|
| `main.py` | Entry point; dispatches train/eval/demo |
| `train_improved.py` | Thin wrapper defaulting to improved MAPPO |
| `environment/grid_env.py` | PettingZoo parallel env (observation, reward, step logic) |
| `environment/model.py` | `ActorCNN` (~587K params), `ActorMLP` (~231K params), centralized `Critic` |
| `environment/train.py` | Baseline MAPPO trainer |
| `environment/train_improved.py` | Enhanced MAPPO (value normalization, Huber loss, configurable critic inputs) |
| `environment/visualize.py` | Training curves, formation traces, animated GIFs |
| `llm/shape_gen.py` | LLM → coordinates; also contains built-in shape generators |
| `cbs_solver.py` | Conflict-Based Search classical planner (comparison baseline) |
| `pi/interactive_display.py` | Raspberry Pi 64×64 RGB LED matrix visualizer (voice-controlled) |

### Observation & Action Space
- **Observation**: 11×11×3 local grid patch + self position + target position + velocity
- **Action**: 9 discrete (stay, N, NE, E, SE, S, SW, W, NW)
- **Actor**: CNN (default) or MLP; both output action logits
- **Critic**: Centralized MLP consuming global state (all agent positions + targets)

### Model Saving
- Baseline models → `models/baseline/`
- Improved models → `models/improved/`
- Naming pattern: `actor_mlp_ep{N}.pt` / `critic_mlp_ep{N}.pt`

# Multi-Agent Formation Control with CTDE

A multi-agent reinforcement learning project where agents learn to form shapes on a 64x64 grid using Centralized Training with Decentralized Execution (CTDE) and LLM-generated target coordinates.

## Project Structure

```
final_project/
├── main.py                  # Main entry point
├── environment/
│   ├── grid_env.py         # PettingZoo grid navigation environment
│   ├── model.py            # Actor-Critic neural networks
│   └── train.py            # MAPPO training algorithm
├── llm/
│   └── shape_gen.py        # LLM-based shape coordinate generation
├── config/                 # Configuration files
├── models/                 # Saved model checkpoints (created during training)
└── requirements.txt        # Python dependencies
```

## Features

- **64x64 Grid Environment**: Agents navigate in 8 directions (cardinal + diagonals)
- **Local Observations**: Each agent observes a local radius around itself (CTDE principle)
- **LLM Shape Generation**: Natural language → coordinates via OpenAI API
- **MAPPO Training**: Multi-Agent Proximal Policy Optimization
- **Collision Avoidance**: Agents are penalized for collisions
- **Formation Rewards**: Bonus rewards for reaching target formation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key (for LLM shape generation):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Training Mode

Train agents to form a circle with 8 agents:
```bash
python main.py --mode train --shape circle --n_agents 8 --n_episodes 1000
```

Train on a custom shape:
```bash
python main.py --mode train --shape "square" --n_agents 4 --n_episodes 500
```

Skip LLM and use default circle formation:
```bash
python main.py --mode train --no_llm --n_agents 6 --n_episodes 1000
```

### Evaluation Mode

Evaluate a trained model:
```bash
python main.py --mode eval --shape circle --n_agents 8
```

Load from specific checkpoint:
```bash
python main.py --mode eval --actor_path models/actor_ep500.pt --critic_path models/critic_ep500.pt
```

### Demo Mode

Run random policy for demonstration:
```bash
python main.py --mode demo --shape triangle --n_agents 6
```

## Command Line Arguments

- `--mode`: Operation mode (`train`, `eval`, `demo`)
- `--n_agents`: Number of agents (default: 4)
- `--shape`: Shape description for LLM (default: 'circle')
- `--n_episodes`: Training episodes (default: 1000)
- `--obs_radius`: Local observation radius (default: 5)
- `--device`: Device to use (`cpu` or `cuda`)
- `--actor_path`: Path to actor checkpoint (for eval mode)
- `--critic_path`: Path to critic checkpoint (for eval mode)
- `--no_llm`: Skip LLM, use default circle formation

## Environment Details

### Observation Space (per agent)
- **local_grid**: (11×11×3) local view with channels: [empty, agents, obstacles]
- **self_position**: (2,) normalized agent position [x, y]
- **target_position**: (2,) normalized target position
- **velocity**: (2,) velocity from last step

### Action Space
9 discrete actions:
- 0: Stay
- 1-8: Move in 8 directions (N, E, S, W, NE, SE, SW, NW)

### Reward Function
```python
reward = -distance_to_target        # Main objective
         - 10 * collision            # Collision penalty
         - 0.01                      # Step penalty
         + 10 * reached_target       # Target bonus
         + 20 * all_at_target        # Formation bonus
```

## Algorithm: MAPPO

**Multi-Agent Proximal Policy Optimization**

- **Centralized Critic**: Uses global state (all agent observations) during training
- **Decentralized Actor**: Each agent uses only local observations during execution
- **PPO Loss**: Clipped surrogate objective with entropy regularization
- **GAE**: Generalized Advantage Estimation for variance reduction

## Neural Network Architecture

### Actor (Decentralized)
```
CNN (3 layers) → Process local grid
MLP → Process state features (position, target, velocity)
Concatenate → FC layers → Action probabilities (9 actions)
```

### Critic (Centralized)
```
Global state (all agents) → MLP (3 layers) → Value estimate
```

## Training Hyperparameters

- Learning rate (actor): 3e-4
- Learning rate (critic): 1e-3
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO epochs: 10
- Clip epsilon: 0.2
- Max gradient norm: 0.5

## Model Checkpoints

Models are saved in `models/` directory:
- Every 100 episodes: `actor_ep{N}.pt`, `critic_ep{N}.pt`
- Final models: `actor_final.pt`, `critic_final.pt`

## Example Output

```
============================================================
Multi-Agent Formation Control with CTDE
============================================================

Step 1: Generating target coordinates for 'circle'...

Generating circle formation for 8 agents on 64x64 grid...

Generated coordinates:
  Agent 0: [32, 48]
  Agent 1: [45, 45]
  Agent 2: [48, 32]
  Agent 3: [45, 19]
  Agent 4: [32, 16]
  Agent 5: [19, 19]
  Agent 6: [16, 32]
  Agent 7: [19, 45]

Step 2: Creating environment...
Environment created with 8 agents

Step 3: Training MAPPO policy...
Training for 1000 episodes...

Episode 10/1000
  Avg Reward: -12.34
  Avg Length: 245.60
  Actor Loss: 0.0234
  Critic Loss: 0.1245
  Entropy: 2.1234

...
```

## Troubleshooting

**Import errors**: Make sure you're running from the project root directory

**OpenAI API errors**: Check your API key is set correctly with `echo $OPENAI_API_KEY`

**Out of memory**: Reduce `--n_agents` or use `--device cpu`

**Training not converging**: Try reducing learning rates or increasing `--n_episodes`

## Future Improvements

- [ ] Add attention mechanisms in Critic for better scalability
- [ ] Implement curriculum learning (start with easier formations)
- [ ] Add communication channels between agents
- [ ] Visualize agent movements with matplotlib/pygame
- [ ] Support for dynamic obstacles
- [ ] Multi-task learning across different shapes

## References

- MAPPO: [https://arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)
- PettingZoo: [https://pettingzoo.farama.org/](https://pettingzoo.farama.org/)
- PPO: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

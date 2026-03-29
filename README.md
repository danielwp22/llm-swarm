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
- **Collision Avoidance**: Agents are penalized for collisions with tracking metrics
- **Formation Rewards**: Bonus rewards for reaching target formation
- **GPU Acceleration**: Automatic CUDA detection for faster training
- **Visualization Tools**: Matplotlib-based plotting and GIF animation generation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For GPU acceleration, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3. Set up OpenAI API key (for LLM shape generation):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### GPU Support

The training automatically detects and uses CUDA if available. You'll see output like:

```
============================================================
Multi-Agent Formation Control with CTDE
============================================================
Device: CUDA
GPU: NVIDIA GeForce RTX 3090
CUDA Version: 11.8
============================================================
```

GPU training can be 10-50x faster than CPU, especially for larger agent counts.

## Usage

### Training Mode

Train agents to form a circle with 8 agents (uses MLP by default, auto-detects GPU):
```bash
python main.py --mode train --shape circle --n_agents 8 --n_episodes 1000
```

Train with CNN architecture:
```bash
python main.py --mode train --shape circle --n_agents 8 --n_episodes 1000 --actor_type cnn
```

Train both architectures for comparison:
```bash
# Train MLP
python main.py --mode train --shape circle --n_agents 8 --n_episodes 1000 --actor_type mlp --visualize

# Train CNN (won't overwrite MLP models)
python main.py --mode train --shape circle --n_agents 8 --n_episodes 1000 --actor_type cnn --visualize
```

Train on a custom shape with explicit GPU usage:
```bash
python main.py --mode train --shape "square" --n_agents 4 --n_episodes 500 --device cuda
```

Train on CPU only:
```bash
python main.py --mode train --shape triangle --n_agents 6 --n_episodes 1000 --device cpu
```

Skip LLM and use default circle formation:
```bash
python main.py --mode train --no_llm --n_agents 6 --n_episodes 1000
```

Train with visualization enabled:
```bash
python main.py --mode train --shape circle --n_agents 8 --n_episodes 500 --visualize
```

### Evaluation Mode

Evaluate a trained MLP model (default):
```bash
python main.py --mode eval --shape circle --n_agents 8 --actor_type mlp
```

Evaluate a trained CNN model:
```bash
python main.py --mode eval --shape circle --n_agents 8 --actor_type cnn
```

Load from specific checkpoint:
```bash
python main.py --mode eval --actor_path models/actor_mlp_ep500.pt --critic_path models/critic_mlp_ep500.pt --actor_type mlp
```

Evaluate with visualization:
```bash
python main.py --mode eval --shape circle --n_agents 8 --actor_type mlp --visualize
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
- `--device`: Device to use (`auto`, `cpu`, or `cuda`; default: 'auto')
  - `auto`: Automatically detects and uses CUDA if available
  - `cuda`: Force GPU usage (requires CUDA)
  - `cpu`: Force CPU usage
- `--actor_type`: Actor architecture (`mlp` or `cnn`; default: 'mlp')
  - `mlp`: MLP-based actor (simpler, fewer parameters, better for sparse data)
  - `cnn`: CNN-based actor (more parameters, better for dense visual patterns)
- `--actor_path`: Path to actor checkpoint (for eval mode; default: `models/actor_{actor_type}_final.pt`)
- `--critic_path`: Path to critic checkpoint (for eval mode; default: `models/critic_{actor_type}_final.pt`)
- `--no_llm`: Skip LLM, use default circle formation
- `--visualize`: Create visualizations (plots and animations)
- `--vis_dir`: Directory to save visualizations (default: 'visualizations')
- `--no_animation`: Skip animation generation (faster, only creates static plots)

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

The project supports two Actor architectures that can be selected with `--actor_type`:

### ActorMLP (Default, Recommended)
```
Flatten local grid (11×11×3 = 363) → Concatenate with state features (6)
→ FC layers (256 → 256 → 128) → Action probabilities (9 actions)

Parameters: ~231K
Best for: Sparse symbolic observations (this task)
```

### ActorCNN (Alternative)
```
CNN (3 layers) → Process local grid
MLP → Process state features (position, target, velocity)
Concatenate → FC layers → Action probabilities (9 actions)

Parameters: ~587K
Best for: Dense visual patterns
```

### Critic (Centralized)
```
Global state (all agents) → MLP (3 layers) → Value estimate

Parameters: ~133K
```

**Architecture Selection:**
- **MLP** is the default and recommended for this task (60% fewer parameters)
- **CNN** available for comparison or if using dense visual observations
- Both architectures can coexist in the same `models/` directory

## Training Hyperparameters

- Learning rate (actor): 3e-4
- Learning rate (critic): 1e-3
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO epochs: 10
- Clip epsilon: 0.2
- Max gradient norm: 0.5

## Training Metrics

The training loop tracks and logs:
- **Average Reward**: Mean reward per agent over last 100 episodes
  - Should increase over time (less negative → positive)
- **Average Episode Length**: Mean number of steps to completion
  - Should decrease as agents learn more efficient paths
- **Average Collisions**: Mean collision count per episode
  - Should decrease over time as agents learn to avoid each other
  - Well-trained agents typically have <2 collisions per episode
- **Actor Loss**: Policy gradient loss
  - Should stabilize after initial training
- **Critic Loss**: Value function loss
  - Should decrease and stabilize
- **Entropy**: Policy entropy (encourages exploration)
  - Should decrease over time as policy becomes more deterministic

## Model Checkpoints

Models are saved in `models/` directory with architecture-specific naming:
- Every 100 episodes: `actor_{actor_type}_ep{N}.pt`, `critic_{actor_type}_ep{N}.pt`
- Final models: `actor_{actor_type}_final.pt`, `critic_{actor_type}_final.pt`

**Examples:**
- MLP: `actor_mlp_final.pt`, `actor_mlp_ep100.pt`, `actor_mlp_ep200.pt`
- CNN: `actor_cnn_final.pt`, `actor_cnn_ep100.pt`, `actor_cnn_ep200.pt`

This allows training and storing both architectures without conflicts.

### Comparing Architectures

To empirically compare MLP vs CNN performance:

```bash
# Train both with same settings
python main.py --mode train --actor_type mlp --n_agents 8 --n_episodes 1000 --visualize --vis_dir results/mlp
python main.py --mode train --actor_type cnn --n_agents 8 --n_episodes 1000 --visualize --vis_dir results/cnn

# Compare the training_rewards_collisions.png plots in both directories
```

**Expected Results:**
- **MLP**: Faster training (fewer parameters), comparable or better final performance
- **CNN**: Slower training (more parameters), may overfit on sparse data
- **For this task**: MLP is recommended due to sparse symbolic observations

## Visualization

The visualization tool creates **five types of outputs** when `--visualize` is enabled:

### Training Metrics Plots (Generated After Training)

#### 1. Comprehensive Metrics Dashboard
6-panel visualization showing all training metrics:
- **Episode Rewards**: Average reward per agent over time
- **Collision Count**: Total collisions per episode
- **Episode Duration**: Number of steps to completion
- **Actor Loss**: Policy gradient loss
- **Critic Loss**: Value function loss
- **Policy Entropy**: Exploration metric

Saved as: `visualizations/training_metrics.png`

#### 2. Key Metrics Focus
2-panel plot emphasizing rewards and collisions:
- Raw values with semi-transparent lines
- Moving average smoothing (automatically scaled window)
- Clear trend visualization

Saved as: `visualizations/training_rewards_collisions.png`

### Formation Visualization Plots

#### 3. Summary Plot
Shows initial, middle, and final states side-by-side:
- Agent trajectories color-coded by agent ID
- Target positions marked with stars
- Grid boundaries and collision markers

Saved as: `visualizations/formation_summary.png`

#### 4. Final Formation Plot
Detailed view of the final formation with:
- Complete agent trajectories
- Final positions and target positions
- Distance indicators between agents and targets

Saved as: `visualizations/final_formation.png`

#### 5. Animated GIF
Full trajectory animation showing:
- Agent movements step-by-step
- Real-time collision detection
- Progress indicator

Saved as: `visualizations/formation_animation.gif`

### Usage Examples

**Train with full visualization (training metrics + formation):**
```bash
python main.py --mode train --shape circle --n_agents 8 --n_episodes 500 --visualize
```

**Train with visualization but skip animation (faster):**
```bash
python main.py --mode train --n_agents 4 --n_episodes 100 --visualize --no_animation
```

**Evaluate existing model (formation visualization only):**
```bash
python main.py --mode eval --n_agents 8 --visualize
```

**Custom output directory:**
```bash
python main.py --mode train --n_agents 4 --n_episodes 100 --visualize --vis_dir my_results
```

### Understanding Training Metrics

**Rewards**: Should trend upward from negative to positive values as agents learn efficient paths to their targets.

**Collisions**: Should decrease over time, indicating improved coordination and collision avoidance. Well-trained agents typically achieve <2 collisions per episode.

**Episode Length**: Should decrease as agents find more direct routes to their targets.

**Losses & Entropy**: Actor/Critic losses should stabilize, while entropy decreases as the policy becomes more deterministic.

---

## Raspberry Pi RGB LED Display

The project includes an interactive voice-controlled visualizer for Raspberry Pi 5 with a 64x64 RGB LED matrix display.

### Hardware Requirements

- Raspberry Pi 5
- 64x64 RGB LED Matrix Panel
- Adafruit RGB Matrix Bonnet
- USB Microphone for voice input
- Speaker (optional, for audio feedback)

### Installation on Raspberry Pi

```bash
# Install Python dependencies
pip3 install torch numpy pillow vosk
pip3 install adafruit_blinka_raspberry_pi5_piomatter

# Download Vosk speech recognition model
cd /path/to/llm_swarm
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### Usage

#### Basic Usage (with voice prompts)
```bash
cd /path/to/llm_swarm
python3 pi/interactive_display.py
```

The script will:
1. Ask you to say a shape name (circle, square, triangle, etc.)
2. Confirm what it heard with yes/no
3. Use default 8 agents (configurable in script)
4. Generate coordinates via LLM
5. Display real-time formation on RGB matrix

#### Configuration

Edit `pi/interactive_display.py` to customize:

```python
# Configuration at top of file
ACTOR_MODEL_PATH = "models/actor_mlp_final.pt"  # Model to use
ACTOR_TYPE = "mlp"  # "mlp" or "cnn"
WIDTH = 64
HEIGHT = 64
STEP_DELAY = 0.1  # Seconds between animation frames
DEFAULT_N_AGENTS = 8  # Default number of agents
SKIP_AGENT_PROMPT = True  # Set to False to always ask for agent count
```

**To enable agent count prompts:**
```python
SKIP_AGENT_PROMPT = False  # Will ask for agent count via voice
```

**To change default agent count:**
```python
DEFAULT_N_AGENTS = 4  # Use 4 agents instead of 8
```

### Display Features

- **Agent Visualization**: Each agent is a 2×2 bright colored square (10 distinct colors)
- **Target Markers**: Dim gray pixels show formation target positions
- **Motion Trails**: 10-step fading trail follows each agent
- **Real-time Updates**: 10 FPS animation (0.1s between steps)
- **Formation Complete**: Display freezes when all agents reach targets

### Color Palette

| Agent | Color   | RGB Value     |
|-------|---------|---------------|
| 0     | Red     | (255, 0, 0)   |
| 1     | Green   | (0, 255, 0)   |
| 2     | Blue    | (0, 0, 255)   |
| 3     | Yellow  | (255, 255, 0) |
| 4     | Magenta | (255, 0, 255) |
| 5     | Cyan    | (0, 255, 255) |
| 6     | Orange  | (255, 128, 0) |
| 7     | Purple  | (128, 0, 255) |
| 8     | White   | (255, 255, 255) |
| 9     | Lime    | (128, 255, 0) |

### Voice Commands

**Shape Selection:**
- Say: "circle", "square", "triangle", "line", "star", etc.
- Confirm with: "yes" or "no"

**Agent Count** (if SKIP_AGENT_PROMPT = False):
- Say number as digit: "4", "8", "12"
- Or say word: "four", "eight", "twelve"
- Confirm with: "yes" or "no"

### Troubleshooting

**Vosk model not found:**
```bash
# Download the model in your llm_swarm directory
cd /path/to/llm_swarm
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

**Audio device errors:**
```bash
# List audio devices
arecord -l

# Update device in script if needed (currently: plughw:2,0)
```

**Model not found:**
Train a model first on your main machine, then copy to Pi:
```bash
# On main machine
python main.py --mode train --actor_type mlp --n_agents 8 --n_episodes 1000

# Copy to Raspberry Pi
scp models/actor_mlp_final.pt pi@raspberrypi:/path/to/llm_swarm/models/
```

**LED matrix not displaying:**
- Check power supply (5V, 4A+ recommended for 64x64 matrix)
- Verify Adafruit RGB Matrix Bonnet connection
- Check jumper settings on bonnet

### Demo Mode

For demonstrations, configure for minimal interaction:
```python
SKIP_AGENT_PROMPT = True
DEFAULT_N_AGENTS = 8
```

This allows quick shape selection without confirming agent count each time.

---

## Example Output

```
============================================================
Multi-Agent Formation Control with CTDE
============================================================
Device: CUDA
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 12.1
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
Starting MAPPO training for 1000 episodes...
Device: cuda
Number of agents: 8
Target coordinates: [[32, 48], [45, 45], [48, 32], [45, 19], [32, 16], [19, 19], [16, 32], [19, 45]]

Episode 10/1000
  Avg Reward: -12.34
  Avg Length: 245.60
  Avg Collisions: 8.30
  Actor Loss: 0.0234
  Critic Loss: 0.1245
  Entropy: 2.1234

...

Models saved at episode 100

Training completed! Final models saved.

============================================================
Running trained policy...
============================================================

Success! All agents reached their targets in 87 steps!

Total steps: 87
Total reward: 142.56
Total collisions: 3
Average reward per agent per step: 0.2046
Collision rate: 0.03 collisions/step

Step 5: Creating visualizations...

Generating training metrics plots...
✓ Training metrics plot saved: visualizations/training_metrics.png
✓ Key metrics plot saved: visualizations/training_rewards_collisions.png

Generating formation visualizations...

Creating visualizations in visualizations/...
✓ Summary plot saved: visualizations/formation_summary.png
✓ Final formation saved: visualizations/final_formation.png
Saving animation to visualizations/formation_animation.gif...
Animation saved successfully!
✓ Animation saved: visualizations/formation_animation.gif

Visualization complete! Files saved in visualizations/
```

## Troubleshooting

**Import errors**: Make sure you're running from the project root directory

**OpenAI API errors**: Check your API key is set correctly with `echo $OPENAI_API_KEY`

**CUDA out of memory**:
- Reduce `--n_agents` (fewer agents = less memory)
- Use `--device cpu` to train on CPU instead
- Reduce batch size or network hidden dimensions in code

**CUDA not detected**:
- Verify installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check NVIDIA drivers: `nvidia-smi`
- Reinstall PyTorch with CUDA support

**Training not converging**:
- Try reducing learning rates in `train.py`
- Increase `--n_episodes`
- Use simpler shapes with fewer agents initially

## Future Improvements

- [ ] Add attention mechanisms in Critic for better scalability
- [ ] Implement curriculum learning (start with easier formations)
- [ ] Add communication channels between agents
- [x] Visualize agent movements with matplotlib (COMPLETED)
- [ ] Support for dynamic obstacles
- [ ] Multi-task learning across different shapes
- [ ] Add tensorboard logging for collision metrics and training curves
- [ ] Implement adaptive collision penalty based on training progress
- [ ] Interactive visualization with real-time agent control

## References

- MAPPO: [https://arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)
- PettingZoo: [https://pettingzoo.farama.org/](https://pettingzoo.farama.org/)
- PPO: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

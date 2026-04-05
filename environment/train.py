import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os

from environment.model import Actor, ActorCNN, ActorMLP, Critic, dict_obs_to_tensor, batch_dict_obs


def _infer_n_agents_from_critic_state_dim(state_dim):
    """
    Infer n_agents from critic input dim where:
    state_dim = n_agents * (n_agents + 6)
    """
    # n^2 + 6n - state_dim = 0
    disc = 36 + 4 * state_dim
    sqrt_disc = int(np.sqrt(disc))
    if sqrt_disc * sqrt_disc != disc:
        return None

    n_agents = (-6 + sqrt_disc) // 2
    if n_agents > 0 and n_agents * (n_agents + 6) == state_dim:
        return int(n_agents)
    return None


def _build_global_state_features(obs_dict, agent_order, n_agents):
    """
    Build centralized critic features in a stable agent order.
    """
    global_state_features = []
    for i, agent in enumerate(agent_order):
        obs = obs_dict[agent]
        self_pos = obs['self_position']
        target_pos = obs['target_position']
        velocity = obs['velocity']
        agent_id = np.zeros(n_agents, dtype=np.float32)
        agent_id[i] = 1.0
        global_state_features.extend([*self_pos, *target_pos, *velocity, *agent_id])
    return global_state_features


class RolloutBuffer:
    """
    Buffer for storing trajectory data during rollout.
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.global_states = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.global_states = []

    def add(self, obs, action, log_prob, reward, done, value, global_state):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.global_states.append(global_state)

    def get(self):
        return (
            self.observations,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
            self.values,
            self.global_states
        )


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, last_value=0.0):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    Returns:
        advantages: Computed advantages
        returns: Discounted returns
    """
    advantages = []
    gae = 0
    next_value = float(last_value)

    # Compute advantages in reverse order
    for t in reversed(range(len(rewards))):
        if t != len(rewards) - 1:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    # Returns are advantages + values
    returns = [adv + val for adv, val in zip(advantages, values)]

    return advantages, returns


def train_mappo(
    env,
    n_agents,
    target_coords,
    target_coords_sampler=None,
    n_episodes=1000,
    max_steps=500,
    obs_radius=5,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epochs=4,
    num_mini_batch=4,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    entropy_coef_end=0.001,
    max_grad_norm=0.5,
    use_clipped_value_loss=True,
    device='cpu',
    save_dir='models',
    log_interval=10,
    actor_type='cnn',
):
    """
    Train multi-agent policy using MAPPO (Multi-Agent PPO).

    Args:
        env: PettingZoo parallel environment
        n_agents: Number of agents
        target_coords: Target coordinates for formation
        target_coords_sampler: Optional callable taking episode index and returning target coordinates
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        obs_radius: Observation radius for agents
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        gae_lambda: GAE lambda
        ppo_epochs: Number of PPO update epochs per rollout
        num_mini_batch: Number of mini-batches per PPO epoch
        clip_epsilon: PPO clipping parameter
        value_loss_coef: Coefficient for value loss
        entropy_coef: Initial coefficient for entropy bonus
        entropy_coef_end: Final entropy coefficient for linear decay
        max_grad_norm: Maximum gradient norm for clipping
        use_clipped_value_loss: Whether to use PPO-style clipped value loss
        device: Device to train on
        save_dir: Directory to save models
        log_interval: Interval for logging progress
        actor_type: Actor architecture ('mlp' or 'cnn')

    Returns:
        actor: Trained actor network
        critic: Trained critic network
        history: Training metrics history
    """
    # Create models
    if actor_type == 'mlp':
        actor = ActorMLP(obs_radius=obs_radius).to(device)
        print(f"Using ActorMLP architecture")
    elif actor_type == 'cnn':
        actor = ActorCNN(obs_radius=obs_radius).to(device)
        print(f"Using ActorCNN architecture")
    else:
        raise ValueError(f"Unknown actor_type: {actor_type}. Choose 'mlp' or 'cnn'")

    critic = Critic(n_agents=n_agents, obs_radius=obs_radius).to(device)

    # Print parameter counts
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")

    # Create optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training metrics (rolling averages for logging)
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_collisions = deque(maxlen=100)

    # Full history for visualization
    history = {
        'rewards': [],
        'lengths': [],
        'collisions': [],
        'actor_loss': [],
        'critic_loss': [],
        'entropy': []
    }

    print(f"Starting MAPPO training for {n_episodes} episodes...")
    print(f"Device: {device}")
    print(f"Number of agents: {n_agents}")
    if target_coords_sampler is None:
        print(f"Target coordinates: {target_coords}\n")
    else:
        print("Target coordinates: sampled per episode\n")

    for episode in range(n_episodes):
        if n_episodes > 1:
            progress = episode / (n_episodes - 1)
            current_entropy_coef = entropy_coef + progress * (entropy_coef_end - entropy_coef)
        else:
            current_entropy_coef = entropy_coef

        current_target_coords = target_coords_sampler(episode) if target_coords_sampler is not None else target_coords

        # Reset environment
        obs, info = env.reset(seed=episode, target_coords=current_target_coords)

        # Rollout buffer for this episode
        buffers = {agent: RolloutBuffer() for agent in env.agents}

        episode_reward = 0
        episode_length = 0
        episode_collision_count = 0
        terminations = {agent: False for agent in env.possible_agents}

        for step in range(max_steps):
            # Collect actions from all agents
            actions = {}

            # Create global state for critic (concatenate all agent observations)
            global_state_features = _build_global_state_features(obs, env.possible_agents, n_agents)

            global_state = torch.FloatTensor(global_state_features).unsqueeze(0).to(device)

            # Get actions and values for each agent
            for agent in env.agents:
                obs_tensor = dict_obs_to_tensor(obs[agent], device)

                with torch.no_grad():
                    action, log_prob = actor.get_action(obs_tensor)
                    value = critic(global_state)

                actions[agent] = action.item()

                # Store in buffer
                buffers[agent].add(
                    obs=obs[agent],
                    action=action.item(),
                    log_prob=log_prob.item(),
                    reward=0,  # Will be filled after step
                    done=False,
                    value=value.item(),
                    global_state=global_state_features
                )

            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Centralized critic predicts one value for the whole team state.
            # Use a shared team reward target for all agents to keep critic targets consistent.
            team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0

            # Update buffers with rewards and dones
            for agent in env.agents:
                if agent in rewards:
                    buffers[agent].rewards[-1] = team_reward
                    # For GAE masks, only true terminations should cut bootstrapping.
                    buffers[agent].dones[-1] = terminations[agent]
                    episode_reward += rewards[agent]
                    # Track collisions
                    if agent in infos and infos[agent].get('collision', False):
                        episode_collision_count += 1

            obs = next_obs
            episode_length += 1

            # Check if episode ended
            if not env.agents or all(terminations.get(agent, False) for agent in env.possible_agents):
                break

        episode_rewards.append(episode_reward / n_agents)  # Average reward per agent
        episode_lengths.append(episode_length)
        episode_collisions.append(episode_collision_count)

        # Store in full history
        history['rewards'].append(episode_reward / n_agents)
        history['lengths'].append(episode_length)
        history['collisions'].append(episode_collision_count)

        # PPO Update
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # Bootstrap value for truncated/non-terminal endings.
        all_terminated = bool(obs) and all(
            terminations.get(agent, False) for agent in env.possible_agents
        )
        if obs and not all_terminated:
            with torch.no_grad():
                next_global_state_features = _build_global_state_features(obs, env.possible_agents, n_agents)
                next_global_state = torch.FloatTensor(next_global_state_features).unsqueeze(0).to(device)
                bootstrap_value = critic(next_global_state).item()
        else:
            bootstrap_value = 0.0

        all_observations = []
        all_actions = []
        all_log_probs_old = []
        all_advantages = []
        all_returns = []
        all_global_states = []
        all_values_old = []

        for _, buffer in buffers.items():
            observations, actions_buf, log_probs_old, rewards_buf, dones_buf, values_buf, global_states = buffer.get()

            if len(observations) == 0:
                continue

            # Compute GAE for this trajectory, then aggregate across agents.
            advantages, returns = compute_gae(
                rewards_buf,
                values_buf,
                dones_buf,
                gamma,
                gae_lambda,
                last_value=bootstrap_value,
            )

            all_observations.extend(observations)
            all_actions.extend(actions_buf)
            all_log_probs_old.extend(log_probs_old)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_global_states.extend(global_states)
            all_values_old.extend(values_buf)

            # Clear buffer
            buffer.clear()

        if len(all_observations) == 0:
            history['actor_loss'].append(0)
            history['critic_loss'].append(0)
            history['entropy'].append(0)
            continue

        obs_batch = batch_dict_obs(all_observations, device)
        actions_tensor = torch.LongTensor(all_actions).to(device)
        log_probs_old_tensor = torch.FloatTensor(all_log_probs_old).to(device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(device)
        returns_tensor = torch.FloatTensor(all_returns).to(device)
        global_states_tensor = torch.FloatTensor(all_global_states).to(device)
        values_old_tensor = torch.FloatTensor(all_values_old).to(device)

        # Normalize advantages over the whole collected batch (MAPPO-style).
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        batch_size = actions_tensor.shape[0]
        mini_batch_size = max(1, batch_size // max(1, num_mini_batch))

        for _ in range(ppo_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                idx = perm[start:end]

                obs_mb = {k: v[idx] for k, v in obs_batch.items()}
                actions_mb = actions_tensor[idx]
                log_probs_old_mb = log_probs_old_tensor[idx]
                adv_mb = advantages_tensor[idx]
                returns_mb = returns_tensor[idx]
                global_states_mb = global_states_tensor[idx]
                values_old_mb = values_old_tensor[idx]

                # Actor loss
                action_probs = actor(obs_mb)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - log_probs_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (optionally clipped to reduce destructive critic steps).
                values_pred = critic(global_states_mb).squeeze(-1)
                if use_clipped_value_loss:
                    values_pred_clipped = values_old_mb + torch.clamp(
                        values_pred - values_old_mb, -clip_epsilon, clip_epsilon
                    )
                    critic_loss_unclipped = (returns_mb - values_pred).pow(2)
                    critic_loss_clipped = (returns_mb - values_pred_clipped).pow(2)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                else:
                    critic_loss = nn.MSELoss()(values_pred, returns_mb)

                loss = actor_loss + value_loss_coef * critic_loss - current_entropy_coef * entropy

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Store losses in history
        denom = n_updates if n_updates > 0 else 1
        history['actor_loss'].append(total_actor_loss / denom)
        history['critic_loss'].append(total_critic_loss / denom)
        history['entropy'].append(total_entropy / denom)

        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_collisions = np.mean(episode_collisions)
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Avg Collisions: {avg_collisions:.2f}")
            print(f"  Actor Loss: {total_actor_loss / denom:.4f}")
            print(f"  Critic Loss: {total_critic_loss / denom:.4f}")
            print(f"  Entropy: {total_entropy / denom:.4f}")
            print(f"  Entropy Coef: {current_entropy_coef:.6f}\n")

        # Save models periodically
        if (episode + 1) % 100 == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f'actor_{actor_type}_ep{episode+1}.pt'))
            torch.save(critic.state_dict(), os.path.join(save_dir, f'critic_{actor_type}_ep{episode+1}.pt'))
            print(f"Models saved at episode {episode + 1}\n")

    # Save final models
    torch.save(actor.state_dict(), os.path.join(save_dir, f'actor_{actor_type}_final.pt'))
    torch.save(critic.state_dict(), os.path.join(save_dir, f'critic_{actor_type}_final.pt'))
    print(f"Training completed! Final models saved to actor_{actor_type}_final.pt and critic_{actor_type}_final.pt\n")

    return actor, critic, history


def load_models(actor_path, critic_path, n_agents, obs_radius=5, device='cpu', actor_type='cnn'):
    """
    Load trained models from checkpoint.

    Args:
        actor_path: Path to actor checkpoint
        critic_path: Path to critic checkpoint
        n_agents: Number of agents
        obs_radius: Observation radius
        device: Device to load models on
        actor_type: Actor architecture ('mlp' or 'cnn')

    Returns:
        actor: Loaded actor network
        critic: Loaded critic network
    """
    if actor_type == 'mlp':
        actor = ActorMLP(obs_radius=obs_radius).to(device)
    elif actor_type == 'cnn':
        actor = ActorCNN(obs_radius=obs_radius).to(device)
    else:
        raise ValueError(f"Unknown actor_type: {actor_type}. Choose 'mlp' or 'cnn'")

    critic = Critic(n_agents=n_agents, obs_radius=obs_radius).to(device)

    actor.load_state_dict(torch.load(actor_path, map_location=device))

    critic_state_dict = torch.load(critic_path, map_location=device)
    try:
        critic.load_state_dict(critic_state_dict)
    except RuntimeError as e:
        # Provide a targeted hint for the common n_agents mismatch case.
        ckpt_fc1 = critic_state_dict.get('fc1.weight')
        inferred_n_agents = None
        if ckpt_fc1 is not None and ckpt_fc1.ndim == 2:
            inferred_n_agents = _infer_n_agents_from_critic_state_dim(ckpt_fc1.shape[1])

        hint = ""
        if inferred_n_agents is not None and inferred_n_agents != n_agents:
            hint = (
                f" Checkpoint critic expects --n_agents {inferred_n_agents}, "
                f"but current run uses --n_agents {n_agents}."
            )
        raise RuntimeError(f"{e}.{hint}") from e

    actor.eval()
    critic.eval()

    print(f"Models loaded from {actor_path} and {critic_path}")

    return actor, critic


def load_actor(actor_path, obs_radius=5, device='cpu', actor_type='cnn'):
    """
    Load only the actor network for decentralized execution (CTDE eval).

    Args:
        actor_path: Path to actor checkpoint
        obs_radius: Observation radius
        device: Device to load model on
        actor_type: Actor architecture ('mlp' or 'cnn')

    Returns:
        actor: Loaded actor network in eval mode
    """
    if actor_type == 'mlp':
        actor = ActorMLP(obs_radius=obs_radius).to(device)
    elif actor_type == 'cnn':
        actor = ActorCNN(obs_radius=obs_radius).to(device)
    else:
        raise ValueError(f"Unknown actor_type: {actor_type}. Choose 'mlp' or 'cnn'")

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print(f"Actor loaded from {actor_path}")
    return actor

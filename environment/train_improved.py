import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.model import ActorCNN, ActorMLP, CentralizedCritic, dict_obs_to_tensor, batch_dict_obs
from environment.train import compute_gae


def _full_local_concat_obs_dim(n_agents, obs_radius):
    local_grid_size = 2 * obs_radius + 1
    grid_dim = local_grid_size * local_grid_size * 3
    per_agent_dim = grid_dim + 8 + n_agents
    return n_agents * per_agent_dim + n_agents


def _build_full_local_concat_obs(obs_dict, agent_order, query_agent, n_agents):
    features = []
    for i, agent in enumerate(agent_order):
        obs = obs_dict[agent]
        local_grid = obs['local_grid'].reshape(-1)
        self_pos = obs['self_position']
        target_pos = obs['target_position']
        velocity = obs['velocity']
        relative_pos = target_pos - self_pos
        agent_id = np.zeros(n_agents, dtype=np.float32)
        agent_id[i] = 1.0
        features.extend(local_grid.tolist())
        features.extend(self_pos.tolist())
        features.extend(target_pos.tolist())
        features.extend(relative_pos.tolist())
        features.extend(velocity.tolist())
        features.extend(agent_id.tolist())

    query_id = np.zeros(n_agents, dtype=np.float32)
    query_id[agent_order.index(query_agent)] = 1.0
    features.extend(query_id.tolist())

    return np.asarray(features, dtype=np.float32)


def _shared_state_obs_dim(n_agents):
    # For each agent: self_pos (2), target_pos (2), velocity (2), relative_pos (2), agent_id (n_agents)
    return n_agents * (8 + n_agents)


def _build_shared_state_obs(obs_dict, agent_order, n_agents):
    features = []
    for i, agent in enumerate(agent_order):
        obs = obs_dict[agent]
        self_pos = obs['self_position']
        target_pos = obs['target_position']
        velocity = obs['velocity']
        relative_pos = target_pos - self_pos
        agent_id = np.zeros(n_agents, dtype=np.float32)
        agent_id[i] = 1.0
        features.extend(self_pos.tolist())
        features.extend(target_pos.tolist())
        features.extend(velocity.tolist())
        features.extend(relative_pos.tolist())
        features.extend(agent_id.tolist())
    return np.asarray(features, dtype=np.float32)


def _agent_specific_obs_dim(n_agents, obs_radius):
    local_grid_size = 2 * obs_radius + 1
    grid_dim = local_grid_size * local_grid_size * 3
    # shared state + query agent local grid + query agent id
    return _shared_state_obs_dim(n_agents) + grid_dim + n_agents


def _build_agent_specific_obs(obs_dict, agent_order, query_agent, n_agents):
    shared_state = _build_shared_state_obs(obs_dict, agent_order, n_agents)
    query_obs = obs_dict[query_agent]
    local_grid = query_obs['local_grid'].reshape(-1)
    query_id = np.zeros(n_agents, dtype=np.float32)
    query_id[agent_order.index(query_agent)] = 1.0
    return np.concatenate([shared_state, local_grid, query_id]).astype(np.float32)


def _critic_obs_builder(critic_input_type, n_agents, obs_radius):
    if critic_input_type == 'shared':
        return _shared_state_obs_dim(n_agents), lambda obs_dict, agent_order, query_agent: _build_shared_state_obs(
            obs_dict, agent_order, n_agents
        )
    if critic_input_type == 'agent_specific':
        return _agent_specific_obs_dim(n_agents, obs_radius), lambda obs_dict, agent_order, query_agent: _build_agent_specific_obs(
            obs_dict, agent_order, query_agent, n_agents
        )
    if critic_input_type == 'full_local_concat':
        return _full_local_concat_obs_dim(n_agents, obs_radius), lambda obs_dict, agent_order, query_agent: _build_full_local_concat_obs(
            obs_dict, agent_order, query_agent, n_agents
        )
    raise ValueError(
        f"Unknown critic_input_type: {critic_input_type}. Choose 'shared', 'agent_specific', or 'full_local_concat'."
    )


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.centralized_obs = []

    def add(self, obs, action, log_prob, reward, done, value, centralized_obs):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.centralized_obs.append(centralized_obs)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.centralized_obs.clear()

    def get(self):
        return (
            self.observations,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
            self.values,
            self.centralized_obs,
        )


class ValueNorm(nn.Module):
    """
    Running normalization for scalar value targets.
    """
    def __init__(self, epsilon=1e-5):
        super(ValueNorm, self).__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.tensor(epsilon))

    def update(self, values):
        values = values.detach().reshape(-1).float()
        if values.numel() == 0:
            return

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = torch.tensor(float(values.numel()), device=values.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1e-6))
        self.count.copy_(total_count)

    def normalize(self, values):
        return (values - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, values):
        return values * torch.sqrt(self.var + 1e-8) + self.mean


def _huber_loss(error, delta=10.0):
    abs_error = error.abs()
    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=error.device))
    linear = abs_error - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


def train_mappo_improved(
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
    ppo_epochs=15,
    num_mini_batch=4,
    clip_epsilon=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    entropy_coef_end=0.001,
    max_grad_norm=10.0,
    use_clipped_value_loss=True,
    use_huber_loss=True,
    huber_delta=10.0,
    use_value_norm=True,
    critic_input_type='agent_specific',
    device='cpu',
    save_dir='models/improved',
    log_interval=10,
    actor_type='cnn',
):
    """
    More MAPPO-like trainer for comparison with the baseline implementation.
    """
    if actor_type == 'mlp':
        actor = ActorMLP(obs_radius=obs_radius).to(device)
        print("Using ActorMLP architecture")
    elif actor_type == 'cnn':
        actor = ActorCNN(obs_radius=obs_radius).to(device)
        print("Using ActorCNN architecture")
    else:
        raise ValueError(f"Unknown actor_type: {actor_type}. Choose 'mlp' or 'cnn'")

    critic_input_dim, build_critic_obs = _critic_obs_builder(critic_input_type, n_agents, obs_radius)
    critic = CentralizedCritic(input_dim=critic_input_dim).to(device)
    value_norm = ValueNorm().to(device) if use_value_norm else None

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    os.makedirs(save_dir, exist_ok=True)

    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_collisions = deque(maxlen=100)

    history = {
        'rewards': [],
        'lengths': [],
        'collisions': [],
        'actor_loss': [],
        'critic_loss': [],
        'entropy': [],
    }

    print(f"Starting improved MAPPO training for {n_episodes} episodes...")
    print(f"Device: {device}")
    print(f"Number of agents: {n_agents}")
    if target_coords_sampler is None:
        print(f"Target coordinates: {target_coords}")
    else:
        print("Target coordinates: sampled per episode")
    print(f"PPO epochs: {ppo_epochs}")
    print(f"Critic Input: {critic_input_type}")
    print(f"Value normalization: {use_value_norm}")
    print(f"Huber critic loss: {use_huber_loss}\n")

    for episode in range(n_episodes):
        if n_episodes > 1:
            progress = episode / (n_episodes - 1)
            current_entropy_coef = entropy_coef + progress * (entropy_coef_end - entropy_coef)
        else:
            current_entropy_coef = entropy_coef

        current_target_coords = target_coords_sampler(episode) if target_coords_sampler is not None else target_coords
        obs, info = env.reset(seed=episode, target_coords=current_target_coords)
        buffers = {agent: RolloutBuffer() for agent in env.agents}

        episode_reward = 0.0
        episode_length = 0
        episode_collision_count = 0
        terminations = {agent: False for agent in env.possible_agents}

        for _ in range(max_steps):
            actions = {}

            for agent in env.agents:
                obs_tensor = dict_obs_to_tensor(obs[agent], device)
                centralized_obs = build_critic_obs(obs, env.possible_agents, agent)
                centralized_obs_tensor = torch.FloatTensor(centralized_obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, log_prob = actor.get_action(obs_tensor)
                    value = critic(centralized_obs_tensor)

                actions[agent] = action.item()

                buffers[agent].add(
                    obs=obs[agent],
                    action=action.item(),
                    log_prob=log_prob.item(),
                    reward=0.0,
                    done=False,
                    value=value.item(),
                    centralized_obs=centralized_obs,
                )

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.possible_agents:
                if agent in rewards:
                    buffers[agent].rewards[-1] = float(rewards[agent])
                    buffers[agent].dones[-1] = bool(terminations[agent])
                    episode_reward += rewards[agent]
                    if infos.get(agent, {}).get('collision', False):
                        episode_collision_count += 1

            obs = next_obs
            episode_length += 1

            if not env.agents or all(terminations.get(agent, False) for agent in env.possible_agents):
                break

        episode_rewards.append(episode_reward / n_agents)
        episode_lengths.append(episode_length)
        episode_collisions.append(episode_collision_count)

        history['rewards'].append(episode_reward / n_agents)
        history['lengths'].append(episode_length)
        history['collisions'].append(episode_collision_count)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        all_observations = []
        all_actions = []
        all_log_probs_old = []
        all_advantages = []
        all_returns = []
        all_centralized_obs = []
        all_values_old = []

        for agent, buffer in buffers.items():
            observations, actions_buf, log_probs_old, rewards_buf, dones_buf, values_buf, centralized_obs_buf = buffer.get()

            if not observations:
                continue

            if obs and not terminations.get(agent, False):
                next_centralized_obs = build_critic_obs(obs, env.possible_agents, agent)
                next_centralized_obs_tensor = torch.FloatTensor(next_centralized_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    bootstrap_value = critic(next_centralized_obs_tensor).item()
                if value_norm is not None:
                    bootstrap_value = float(value_norm.denormalize(torch.tensor([bootstrap_value], device=device)).item())
            else:
                bootstrap_value = 0.0

            if value_norm is not None:
                values_for_gae = value_norm.denormalize(
                    torch.tensor(values_buf, dtype=torch.float32, device=device)
                ).cpu().numpy().tolist()
            else:
                values_for_gae = values_buf

            advantages, returns = compute_gae(
                rewards_buf,
                values_for_gae,
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
            all_centralized_obs.extend(centralized_obs_buf)
            all_values_old.extend(values_buf)

            buffer.clear()

        if not all_observations:
            history['actor_loss'].append(0.0)
            history['critic_loss'].append(0.0)
            history['entropy'].append(0.0)
            continue

        obs_batch = batch_dict_obs(all_observations, device)
        actions_tensor = torch.LongTensor(all_actions).to(device)
        log_probs_old_tensor = torch.FloatTensor(all_log_probs_old).to(device)
        returns_tensor = torch.FloatTensor(all_returns).to(device)
        centralized_obs_tensor = torch.FloatTensor(np.array(all_centralized_obs)).to(device)
        values_old_tensor = torch.FloatTensor(all_values_old).to(device)

        if value_norm is not None:
            values_old_for_adv = value_norm.denormalize(values_old_tensor)
            advantages_tensor = returns_tensor - values_old_for_adv
        else:
            advantages_tensor = returns_tensor - values_old_tensor

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)

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
                centralized_obs_mb = centralized_obs_tensor[idx]
                values_old_mb = values_old_tensor[idx]

                action_probs = actor(obs_mb)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - log_probs_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = critic(centralized_obs_mb).squeeze(-1)
                target_values = returns_mb
                if value_norm is not None:
                    value_norm.update(returns_mb)
                    target_values = value_norm.normalize(returns_mb)

                if use_clipped_value_loss:
                    values_pred_clipped = values_old_mb + torch.clamp(
                        values_pred - values_old_mb, -clip_epsilon, clip_epsilon
                    )
                    error_original = target_values - values_pred
                    error_clipped = target_values - values_pred_clipped
                    if use_huber_loss:
                        critic_loss_original = _huber_loss(error_original, delta=huber_delta)
                        critic_loss_clipped = _huber_loss(error_clipped, delta=huber_delta)
                    else:
                        critic_loss_original = error_original.pow(2)
                        critic_loss_clipped = error_clipped.pow(2)
                    critic_loss = torch.max(critic_loss_original, critic_loss_clipped).mean()
                else:
                    error = target_values - values_pred
                    critic_loss = _huber_loss(error, delta=huber_delta).mean() if use_huber_loss else error.pow(2).mean()

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                (actor_loss - current_entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optimizer.step()

                (critic_loss * value_loss_coef).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        denom = n_updates if n_updates > 0 else 1
        history['actor_loss'].append(total_actor_loss / denom)
        history['critic_loss'].append(total_critic_loss / denom)
        history['entropy'].append(total_entropy / denom)

        if (episode + 1) % log_interval == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {np.mean(episode_rewards):.2f}")
            print(f"  Avg Length: {np.mean(episode_lengths):.2f}")
            print(f"  Avg Collisions: {np.mean(episode_collisions):.2f}")
            print(f"  Actor Loss: {total_actor_loss / denom:.4f}")
            print(f"  Critic Loss: {total_critic_loss / denom:.4f}")
            print(f"  Entropy: {total_entropy / denom:.4f}")
            print(f"  Entropy Coef: {current_entropy_coef:.6f}\n")

        if (episode + 1) % 100 == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f'actor_{actor_type}_ep{episode+1}.pt'))
            torch.save(critic.state_dict(), os.path.join(save_dir, f'critic_{actor_type}_ep{episode+1}.pt'))
            print(f"Models saved at episode {episode + 1}\n")

    torch.save(actor.state_dict(), os.path.join(save_dir, f'actor_{actor_type}_final.pt'))
    torch.save(critic.state_dict(), os.path.join(save_dir, f'critic_{actor_type}_final.pt'))
    print(f"Training completed! Final models saved to {save_dir}\n")

    return actor, critic, history

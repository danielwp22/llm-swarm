import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os

from environment.model import Actor, Critic, dict_obs_to_tensor, batch_dict_obs


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


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
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
    next_value = 0

    # Compute advantages in reverse order
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
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
    n_episodes=1000,
    max_steps=500,
    obs_radius=5,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epochs=10,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    device='cpu',
    save_dir='models',
    log_interval=10,
):
    """
    Train multi-agent policy using MAPPO (Multi-Agent PPO).

    Args:
        env: PettingZoo parallel environment
        n_agents: Number of agents
        target_coords: Target coordinates for formation
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        obs_radius: Observation radius for agents
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        gae_lambda: GAE lambda
        ppo_epochs: Number of PPO update epochs per rollout
        clip_epsilon: PPO clipping parameter
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to train on
        save_dir: Directory to save models
        log_interval: Interval for logging progress

    Returns:
        actor: Trained actor network
        critic: Trained critic network
    """
    # Create models
    actor = Actor(obs_radius=obs_radius).to(device)
    critic = Critic(n_agents=n_agents, obs_radius=obs_radius).to(device)

    # Create optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    print(f"Starting MAPPO training for {n_episodes} episodes...")
    print(f"Device: {device}")
    print(f"Number of agents: {n_agents}")
    print(f"Target coordinates: {target_coords}\n")

    for episode in range(n_episodes):
        # Reset environment
        obs, info = env.reset(seed=episode, target_coords=target_coords)

        # Rollout buffer for this episode
        buffers = {agent: RolloutBuffer() for agent in env.agents}

        episode_reward = 0
        episode_length = 0

        for step in range(max_steps):
            # Collect actions from all agents
            actions = {}
            log_probs = {}

            # Create global state for critic (concatenate all agent observations)
            global_state_features = []
            for i, agent in enumerate(env.agents):
                obs_tensor = dict_obs_to_tensor(obs[agent], device)

                # Get features for global state
                self_pos = obs[agent]['self_position']
                target_pos = obs[agent]['target_position']
                velocity = obs[agent]['velocity']
                agent_id = np.zeros(n_agents)
                agent_id[i] = 1.0
                global_state_features.extend([*self_pos, *target_pos, *velocity, *agent_id])

            global_state = torch.FloatTensor(global_state_features).unsqueeze(0).to(device)

            # Get actions and values for each agent
            agent_obs_batch = []
            for agent in env.agents:
                obs_tensor = dict_obs_to_tensor(obs[agent], device)
                agent_obs_batch.append(obs[agent])

                with torch.no_grad():
                    action, log_prob = actor.get_action(obs_tensor)
                    value = critic(global_state)

                actions[agent] = action.item()
                log_probs[agent] = log_prob.item()

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

            # Update buffers with rewards and dones
            for agent in env.agents:
                if agent in rewards:
                    buffers[agent].rewards[-1] = rewards[agent]
                    buffers[agent].dones[-1] = terminations[agent] or truncations[agent]
                    episode_reward += rewards[agent]

            obs = next_obs
            episode_length += 1

            # Check if episode ended
            if not env.agents or all(terminations.get(agent, False) for agent in env.possible_agents):
                break

        episode_rewards.append(episode_reward / n_agents)  # Average reward per agent
        episode_lengths.append(episode_length)

        # PPO Update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for agent_name, buffer in buffers.items():
            observations, actions_buf, log_probs_old, rewards_buf, dones_buf, values_buf, global_states = buffer.get()

            if len(observations) == 0:
                continue

            # Compute GAE
            advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, gamma, gae_lambda)

            # Convert to tensors
            obs_batch = batch_dict_obs(observations, device)
            actions_tensor = torch.LongTensor(actions_buf).to(device)
            log_probs_old_tensor = torch.FloatTensor(log_probs_old).to(device)
            advantages_tensor = torch.FloatTensor(advantages).to(device)
            returns_tensor = torch.FloatTensor(returns).to(device)
            global_states_tensor = torch.FloatTensor(global_states).to(device)

            # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            # PPO update epochs
            for _ in range(ppo_epochs):
                # Actor loss
                action_probs = actor(obs_batch)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()

                # PPO clipped loss
                ratio = torch.exp(log_probs_new - log_probs_old_tensor)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values_pred = critic(global_states_tensor).squeeze()
                critic_loss = nn.MSELoss()(values_pred, returns_tensor)

                # Total loss
                loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

                # Update networks
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

            # Clear buffer
            buffer.clear()

        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Actor Loss: {total_actor_loss / (n_agents * ppo_epochs):.4f}")
            print(f"  Critic Loss: {total_critic_loss / (n_agents * ppo_epochs):.4f}")
            print(f"  Entropy: {total_entropy / (n_agents * ppo_epochs):.4f}\n")

        # Save models periodically
        if (episode + 1) % 100 == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f'actor_ep{episode+1}.pt'))
            torch.save(critic.state_dict(), os.path.join(save_dir, f'critic_ep{episode+1}.pt'))
            print(f"Models saved at episode {episode + 1}\n")

    # Save final models
    torch.save(actor.state_dict(), os.path.join(save_dir, 'actor_final.pt'))
    torch.save(critic.state_dict(), os.path.join(save_dir, 'critic_final.pt'))
    print("Training completed! Final models saved.\n")

    return actor, critic


def load_models(actor_path, critic_path, n_agents, obs_radius=5, device='cpu'):
    """
    Load trained models from checkpoint.

    Args:
        actor_path: Path to actor checkpoint
        critic_path: Path to critic checkpoint
        n_agents: Number of agents
        obs_radius: Observation radius
        device: Device to load models on

    Returns:
        actor: Loaded actor network
        critic: Loaded critic network
    """
    actor = Actor(obs_radius=obs_radius).to(device)
    critic = Critic(n_agents=n_agents, obs_radius=obs_radius).to(device)

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    critic.load_state_dict(torch.load(critic_path, map_location=device))

    actor.eval()
    critic.eval()

    print(f"Models loaded from {actor_path} and {critic_path}")

    return actor, critic

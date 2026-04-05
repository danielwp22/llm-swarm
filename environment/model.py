import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCNN(nn.Module):
    """
    CNN-based Decentralized Actor network for MAPPO.
    Uses convolutional layers to process spatial local grid.
    Takes local observations and outputs action probabilities.
    """
    def __init__(self, obs_radius=5, hidden_dim=128):
        super(ActorCNN, self).__init__()

        self.obs_radius = obs_radius
        local_grid_size = 2 * obs_radius + 1

        # CNN for processing local grid observations (11x11x3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate flattened conv output size
        conv_output_size = local_grid_size * local_grid_size * 64

        # MLP for position, target, relative position, and velocity (2 + 2 + 2 + 2 = 8 dimensions)
        self.fc_state = nn.Linear(8, 64)

        # Combine CNN output with state features
        self.fc1 = nn.Linear(conv_output_size + 64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Action head (9 actions: stay + 8 directions)
        self.action_head = nn.Linear(hidden_dim, 9)

    def forward(self, obs):
        """
        Args:
            obs: Dict with keys 'local_grid', 'self_position', 'target_position', 'velocity'
        Returns:
            action_probs: Probability distribution over actions
        """
        # Process local grid through CNN
        local_grid = obs['local_grid']  # (batch, 11, 11, 3)
        local_grid = local_grid.permute(0, 3, 1, 2)  # (batch, 3, 11, 11)

        x_conv = F.relu(self.conv1(local_grid))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))
        x_conv = x_conv.flatten(1)  # (batch, conv_output_size)

        # Process state features
        self_pos = obs['self_position']  # (batch, 2)
        target_pos = obs['target_position']  # (batch, 2)
        velocity = obs['velocity']  # (batch, 2)
        relative_pos = target_pos - self_pos  # (batch, 2) explicit direction to goal
        state_features = torch.cat([self_pos, target_pos, relative_pos, velocity], dim=1)  # (batch, 8)

        x_state = F.relu(self.fc_state(state_features))

        # Combine features
        x = torch.cat([x_conv, x_state], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output action probabilities
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs

    def get_action(self, obs, deterministic=False):
        """
        Sample an action from the policy.

        Args:
            obs: Observation dict
            deterministic: If True, return argmax action
        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        action_probs = self.forward(obs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


class ActorMLP(nn.Module):
    """
    MLP-based Decentralized Actor network for MAPPO.
    Flattens local grid and processes with fully connected layers.
    Simpler architecture that may work better for sparse symbolic observations.
    """
    def __init__(self, obs_radius=5, hidden_dim=256):
        super(ActorMLP, self).__init__()

        self.obs_radius = obs_radius
        local_grid_size = 2 * obs_radius + 1

        # Flatten local grid: (11, 11, 3) = 363 dimensions
        grid_input_size = local_grid_size * local_grid_size * 3

        # State features: position (2) + target (2) + relative position (2) + velocity (2) = 8 dimensions
        state_input_size = 8

        # Total input size
        total_input_size = grid_input_size + state_input_size  # 363 + 8 = 371

        # MLP layers
        self.fc1 = nn.Linear(total_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Action head (9 actions: stay + 8 directions)
        self.action_head = nn.Linear(hidden_dim // 2, 9)

    def forward(self, obs):
        """
        Args:
            obs: Dict with keys 'local_grid', 'self_position', 'target_position', 'velocity'
        Returns:
            action_probs: Probability distribution over actions
        """
        # Flatten local grid
        local_grid = obs['local_grid']  # (batch, 11, 11, 3)
        grid_flat = local_grid.flatten(1)  # (batch, 363)

        # Concatenate state features
        self_pos = obs['self_position']  # (batch, 2)
        target_pos = obs['target_position']  # (batch, 2)
        velocity = obs['velocity']  # (batch, 2)
        relative_pos = target_pos - self_pos  # (batch, 2) explicit direction to goal
        state_features = torch.cat([self_pos, target_pos, relative_pos, velocity], dim=1)  # (batch, 8)

        # Combine all features
        x = torch.cat([grid_flat, state_features], dim=1)  # (batch, 371)

        # Process through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output action probabilities
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs

    def get_action(self, obs, deterministic=False):
        """
        Sample an action from the policy.

        Args:
            obs: Observation dict
            deterministic: If True, return argmax action
        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        action_probs = self.forward(obs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


# Default Actor is CNN
Actor = ActorCNN


class Critic(nn.Module):
    """
    Centralized Critic network for MAPPO.
    Takes global state (all agent observations) and outputs state value.
    """
    def __init__(self, n_agents, obs_radius=5, hidden_dim=256):
        super(Critic, self).__init__()

        self.n_agents = n_agents
        self.obs_radius = obs_radius
        local_grid_size = 2 * obs_radius + 1

        # Process each agent's observations
        # For simplicity, we'll concatenate all agent features
        # In a more sophisticated version, you could use attention mechanisms

        # Each agent contributes: position (2) + target (2) + velocity (2) + one-hot ID (n_agents)
        # Total: n_agents * (6 + n_agents)
        total_state_dim = n_agents * (6 + n_agents)

        # MLP for global state
        self.fc1 = nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_obs):
        """
        Args:
            global_obs: List of observation dicts for all agents
                        or a batched version (batch, n_agents, obs_dim)
        Returns:
            value: State value estimate
        """
        # Concatenate all agent states
        if isinstance(global_obs, list):
            # List of dicts -> extract features
            features = []
            for i, obs in enumerate(global_obs):
                self_pos = obs['self_position']
                target_pos = obs['target_position']
                velocity = obs['velocity']
                agent_id = torch.zeros(len(global_obs))
                agent_id[i] = 1.0
                features.append(torch.cat([self_pos, target_pos, velocity, agent_id]))
            global_state = torch.cat(features, dim=0)
        else:
            # Assume already batched and concatenated
            global_state = global_obs

        # Process through MLP
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value_head(x)

        return value


class CentralizedCritic(nn.Module):
    """
    Centralized critic with configurable input size for richer joint observations.
    """
    def __init__(self, input_dim, hidden_dim=512):
        super(CentralizedCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_head(x)


def dict_obs_to_tensor(obs_dict, device='cpu'):
    """
    Convert observation dict to tensors on specified device.

    Args:
        obs_dict: Dict with keys 'local_grid', 'self_position', 'target_position', 'velocity'
        device: Device to place tensors on
    Returns:
        Tensors on device
    """
    return {
        'local_grid': torch.FloatTensor(obs_dict['local_grid']).unsqueeze(0).to(device),
        'self_position': torch.FloatTensor(obs_dict['self_position']).unsqueeze(0).to(device),
        'target_position': torch.FloatTensor(obs_dict['target_position']).unsqueeze(0).to(device),
        'velocity': torch.FloatTensor(obs_dict['velocity']).unsqueeze(0).to(device),
    }


def batch_dict_obs(obs_list, device='cpu'):
    """
    Batch a list of observation dicts.

    Args:
        obs_list: List of observation dicts
        device: Device to place tensors on
    Returns:
        Batched observation dict
    """
    return {
        'local_grid': torch.FloatTensor(np.array([obs['local_grid'] for obs in obs_list])).to(device),
        'self_position': torch.FloatTensor(np.array([obs['self_position'] for obs in obs_list])).to(device),
        'target_position': torch.FloatTensor(np.array([obs['target_position'] for obs in obs_list])).to(device),
        'velocity': torch.FloatTensor(np.array([obs['velocity'] for obs in obs_list])).to(device),
    }


if __name__ == "__main__":
    # Test the models
    obs_radius = 5
    n_agents = 4
    batch_size = 8

    # Create dummy observation
    local_grid_size = 2 * obs_radius + 1
    dummy_obs = {
        'local_grid': torch.randn(batch_size, local_grid_size, local_grid_size, 3),
        'self_position': torch.randn(batch_size, 2),
        'target_position': torch.randn(batch_size, 2),
        'velocity': torch.randn(batch_size, 2),
    }

    # Test ActorMLP
    actor_mlp = ActorMLP(obs_radius=obs_radius)
    action_probs = actor_mlp(dummy_obs)
    print(f"ActorMLP output shape: {action_probs.shape}")  # Should be (batch_size, 9)
    print(f"ActorMLP action probs sum: {action_probs[0].sum()}")  # Should be ~1.0
    print(f"ActorMLP parameters: {sum(p.numel() for p in actor_mlp.parameters())}")

    # Test ActorCNN
    actor_cnn = ActorCNN(obs_radius=obs_radius)
    action_probs = actor_cnn(dummy_obs)
    print(f"\nActorCNN output shape: {action_probs.shape}")  # Should be (batch_size, 9)
    print(f"ActorCNN action probs sum: {action_probs[0].sum()}")  # Should be ~1.0
    print(f"ActorCNN parameters: {sum(p.numel() for p in actor_cnn.parameters())}")

    # Test Critic
    critic = Critic(n_agents=n_agents, obs_radius=obs_radius)
    global_state = torch.randn(batch_size, n_agents * (6 + n_agents))
    value = critic(global_state)
    print(f"Critic output shape: {value.shape}")  # Should be (batch_size, 1)

    print("\nModels initialized successfully!")

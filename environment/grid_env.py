import functools
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


# Action mappings for 8-directional movement + stay
ACTIONS = {
    0: (0, 0),    # Stay
    1: (0, 1),    # North
    2: (1, 0),    # East
    3: (0, -1),   # South
    4: (-1, 0),   # West
    5: (1, 1),    # Northeast
    6: (1, -1),   # Southeast
    7: (-1, -1),  # Southwest
    8: (-1, 1),   # Northwest
}

GRID_SIZE = 64
MAX_STEPS = 500


def env(render_mode=None):
    """Wraps the environment with useful wrappers."""
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env_instance = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env_instance = wrappers.CaptureStdoutWrapper(env_instance)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def raw_env(render_mode=None):
    """Converts ParallelEnv to AEC API."""
    env_instance = parallel_env(render_mode=render_mode)
    env_instance = parallel_to_aec(env_instance)
    return env_instance


class parallel_env(ParallelEnv):
    """
    Multi-agent grid navigation environment for formation control.
    Agents navigate a 64x64 grid to reach target coordinates while avoiding collisions.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "grid_formation_v1"}

    def __init__(self, render_mode=None, n_agents=4, obs_radius=5):
        """
        Args:
            render_mode: Visualization mode
            n_agents: Number of agents in the environment
            obs_radius: Radius of local observation for each agent
        """
        self.n_agents = n_agents
        self.obs_radius = obs_radius
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS

        # Agent names
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))
        self.render_mode = render_mode

        # Environment state
        self.agent_positions = {}
        self.target_positions = {}
        self.step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Observation includes:
        - Local grid view (2*radius+1 x 2*radius+1 x 3): [empty, agents, obstacles]
        - Self position (normalized): [x, y]
        - Target position (normalized): [x, y]
        - Velocity from last step: [dx, dy]
        """
        local_grid_size = 2 * self.obs_radius + 1
        return Dict({
            'local_grid': Box(low=0, high=1, shape=(local_grid_size, local_grid_size, 3), dtype=np.float32),
            'self_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'velocity': Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """9 actions: stay + 8 directions"""
        return Discrete(9)

    def reset(self, seed=None, options=None, target_coords=None):
        """
        Reset environment with random agent positions and specified target coordinates.

        Args:
            target_coords: List of [x, y] coordinates for each agent's target position
        """
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Initialize random positions for agents (ensure no overlaps)
        self.agent_positions = {}
        occupied = set()
        for agent in self.agents:
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in occupied:
                    self.agent_positions[agent] = np.array(pos, dtype=np.float32)
                    occupied.add(pos)
                    break

        # Set target positions
        if target_coords is None:
            # Default: random targets
            target_coords = []
            for _ in range(self.n_agents):
                while True:
                    pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                    if pos not in occupied:
                        target_coords.append(list(pos))
                        occupied.add(pos)
                        break

        self.target_positions = {
            agent: np.array(target_coords[i], dtype=np.float32)
            for i, agent in enumerate(self.agents)
        }

        # Initialize velocities
        self.velocities = {agent: np.array([0, 0], dtype=np.float32) for agent in self.agents}
        self.arrival_bonus_awarded = {agent: False for agent in self.agents}


        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _get_observation(self, agent):
        """Get local observation for an agent."""
        pos = self.agent_positions[agent]
        target = self.target_positions[agent]

        # Create local grid view
        local_grid = self._get_local_grid(agent)

        # Normalize positions to [0, 1]
        self_pos_norm = pos / self.grid_size
        target_pos_norm = target / self.grid_size

        # Get velocity (normalized)
        velocity = self.velocities[agent]

        return {
            'local_grid': local_grid,
            'self_position': self_pos_norm,
            'target_position': target_pos_norm,
            'velocity': velocity,
        }

    def _get_local_grid(self, agent):
        """
        Extract local grid view around agent.
        Returns (2*radius+1, 2*radius+1, 3) array with channels:
        - Channel 0: Empty space
        - Channel 1: Other agents
        - Channel 2: Obstacles (walls/boundaries)
        """
        pos = self.agent_positions[agent].astype(int)
        local_size = 2 * self.obs_radius + 1
        local_grid = np.zeros((local_size, local_size, 3), dtype=np.float32)

        # Mark empty space
        local_grid[:, :, 0] = 1.0

        for i in range(local_size):
            for j in range(local_size):
                # Calculate global position
                global_x = pos[0] + (i - self.obs_radius)
                global_y = pos[1] + (j - self.obs_radius)

                # Check boundaries (mark as obstacles)
                if global_x < 0 or global_x >= self.grid_size or global_y < 0 or global_y >= self.grid_size:
                    local_grid[i, j, 0] = 0
                    local_grid[i, j, 2] = 1  # Obstacle
                    continue

                # Check for other agents
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_pos = self.agent_positions[other_agent].astype(int)
                        if other_pos[0] == global_x and other_pos[1] == global_y:
                            local_grid[i, j, 0] = 0
                            local_grid[i, j, 1] = 1  # Other agent

        return local_grid

    def step(self, actions):
        """Execute actions for all agents."""
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Store old positions for collision detection
        old_positions = {agent: self.agent_positions[agent].copy() for agent in self.agents}
        new_positions = {}

        # Calculate new positions
        for agent in self.agents:
            action = actions[agent]
            delta = np.array(ACTIONS[action], dtype=np.float32)
            new_pos = old_positions[agent] + delta

            # Clip to grid boundaries
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_positions[agent] = new_pos

            # Update velocity
            self.velocities[agent] = new_pos - old_positions[agent]

        # Check for collisions and resolve
        collision_flags = {agent: False for agent in self.agents}
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                if np.array_equal(new_positions[agent1], new_positions[agent2]):
                    # Collision detected - revert both agents to old positions
                    new_positions[agent1] = old_positions[agent1]
                    new_positions[agent2] = old_positions[agent2]
                    collision_flags[agent1] = True
                    collision_flags[agent2] = True

        # Update positions
        self.agent_positions = new_positions

        # Calculate rewards
        rewards = {}
        for agent in self.agents:
            distance = np.linalg.norm(self.agent_positions[agent] - self.target_positions[agent])

            distance_reward = -distance / (self.grid_size / 10)  # ~10x stronger penalty
            collision_penalty = -0.5 if collision_flags[agent] else 0.0
            step_penalty = -0.01  # Encourage faster convergence

            # Pay the arrival bonus only the first time an agent reaches its target.
            just_arrived = distance < 2.5 and not self.arrival_bonus_awarded[agent]
            at_target_bonus = 10.0 if just_arrived else 0.0
            if just_arrived:
                self.arrival_bonus_awarded[agent] = True

            rewards[agent] = distance_reward + collision_penalty + step_penalty + at_target_bonus

        # Check if all agents reached targets
        all_at_target = all(
            np.linalg.norm(self.agent_positions[agent] - self.target_positions[agent]) < 2.5
            for agent in self.agents
        )

        # Add formation bonus if all agents reached targets
        if all_at_target:
            for agent in self.agents:
                rewards[agent] += 20.0

        self.step_count += 1

        # Termination conditions
        terminations = {agent: all_at_target for agent in self.agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {'collision': collision_flags[agent]} for agent in self.agents}

        # Remove agents if episode ended
        if all_at_target or self.step_count >= self.max_steps:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render the environment state."""
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            for agent in self.possible_agents:
                if agent in self.agent_positions:
                    pos = self.agent_positions[agent]
                    target = self.target_positions[agent]
                    dist = np.linalg.norm(pos - target)
                    print(f"{agent}: pos=({pos[0]:.1f}, {pos[1]:.1f}), "
                          f"target=({target[0]:.1f}, {target[1]:.1f}), dist={dist:.2f}")

    def close(self):
        """Cleanup resources."""
        pass

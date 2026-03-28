import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import torch


class GridVisualizer:
    """
    Visualizer for multi-agent grid formation control.
    Displays agent positions, targets, and trajectories on a 2D grid.
    """

    def __init__(self, grid_size=64, n_agents=4, figsize=(10, 10)):
        """
        Args:
            grid_size: Size of the grid
            n_agents: Number of agents
            figsize: Figure size for matplotlib
        """
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.figsize = figsize

        # Color map for agents
        self.colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

        # Storage for trajectory data
        self.reset_trajectory()

    def reset_trajectory(self):
        """Reset trajectory storage."""
        self.trajectories = {i: [] for i in range(self.n_agents)}
        self.target_positions = None
        self.collision_steps = []

    def add_step(self, agent_positions, target_positions, collisions=None):
        """
        Add a step to the trajectory.

        Args:
            agent_positions: Dict or list of agent positions [[x1, y1], [x2, y2], ...]
            target_positions: Dict or list of target positions
            collisions: List of agent indices that collided this step
        """
        if self.target_positions is None:
            if isinstance(target_positions, dict):
                self.target_positions = [target_positions[f"agent_{i}"] for i in range(self.n_agents)]
            else:
                self.target_positions = target_positions

        # Convert to list if dict
        if isinstance(agent_positions, dict):
            positions = [agent_positions[f"agent_{i}"] for i in range(self.n_agents)]
        else:
            positions = agent_positions

        # Store positions
        for i, pos in enumerate(positions):
            self.trajectories[i].append(pos)

        # Store collision info
        if collisions:
            self.collision_steps.append(len(self.trajectories[0]) - 1)

    def plot_step(self, step_idx, save_path=None, show=True):
        """
        Plot a single step.

        Args:
            step_idx: Step index to plot
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Set up grid
        ax.set_xlim(-2, self.grid_size + 2)
        ax.set_ylim(-2, self.grid_size + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(f'Multi-Agent Formation Control - Step {step_idx}', fontsize=14, fontweight='bold')

        # Draw grid boundaries
        boundary = patches.Rectangle((0, 0), self.grid_size, self.grid_size,
                                     linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(boundary)

        # Plot trajectories up to this step
        for agent_idx in range(self.n_agents):
            if step_idx < len(self.trajectories[agent_idx]):
                traj = np.array(self.trajectories[agent_idx][:step_idx+1])
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=self.colors[agent_idx],
                           alpha=0.5, linewidth=2, label=f'Agent {agent_idx} path')

        # Plot current agent positions
        for agent_idx in range(self.n_agents):
            if step_idx < len(self.trajectories[agent_idx]):
                pos = self.trajectories[agent_idx][step_idx]
                ax.scatter(pos[0], pos[1], s=300, c=[self.colors[agent_idx]],
                          edgecolors='black', linewidths=2, marker='o',
                          label=f'Agent {agent_idx}', zorder=5)
                # Add agent number
                ax.text(pos[0], pos[1], str(agent_idx), ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white', zorder=6)

        # Plot target positions
        for agent_idx, target in enumerate(self.target_positions):
            ax.scatter(target[0], target[1], s=300, c=[self.colors[agent_idx]],
                      marker='*', edgecolors='black', linewidths=2,
                      label=f'Target {agent_idx}', zorder=4, alpha=0.6)
            # Draw line from agent to target
            if step_idx < len(self.trajectories[agent_idx]):
                current_pos = self.trajectories[agent_idx][step_idx]
                ax.plot([current_pos[0], target[0]], [current_pos[1], target[1]],
                       '--', color=self.colors[agent_idx], alpha=0.3, linewidth=1)

        # Highlight collisions
        if step_idx in self.collision_steps:
            ax.text(self.grid_size / 2, -1, 'COLLISION!', ha='center',
                   fontsize=14, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        # Legend (simplified to avoid clutter)
        handles, labels = ax.get_legend_handles_labels()
        # Only show agent and target labels, skip trajectory labels
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if 'path' not in l:
                filtered_handles.append(h)
                filtered_labels.append(l)
        ax.legend(filtered_handles, filtered_labels, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    def create_animation(self, save_path='formation_animation.gif', fps=10, show_final=True):
        """
        Create an animation of the entire trajectory.

        Args:
            save_path: Path to save animation
            fps: Frames per second
            show_final: Whether to show the final frame
        """
        if not self.trajectories[0]:
            print("No trajectory data to animate")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        max_steps = len(self.trajectories[0])

        def init():
            ax.clear()
            ax.set_xlim(-2, self.grid_size + 2)
            ax.set_ylim(-2, self.grid_size + 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)

            # Draw grid boundaries
            boundary = patches.Rectangle((0, 0), self.grid_size, self.grid_size,
                                         linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(boundary)

            return []

        def update(frame):
            ax.clear()
            ax.set_xlim(-2, self.grid_size + 2)
            ax.set_ylim(-2, self.grid_size + 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)
            ax.set_title(f'Multi-Agent Formation Control - Step {frame}/{max_steps-1}',
                        fontsize=14, fontweight='bold')

            # Draw grid boundaries
            boundary = patches.Rectangle((0, 0), self.grid_size, self.grid_size,
                                         linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(boundary)

            # Plot trajectories up to current frame
            for agent_idx in range(self.n_agents):
                traj = np.array(self.trajectories[agent_idx][:frame+1])
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=self.colors[agent_idx],
                           alpha=0.5, linewidth=2)

            # Plot current agent positions
            for agent_idx in range(self.n_agents):
                pos = self.trajectories[agent_idx][frame]
                ax.scatter(pos[0], pos[1], s=300, c=[self.colors[agent_idx]],
                          edgecolors='black', linewidths=2, marker='o', zorder=5)
                ax.text(pos[0], pos[1], str(agent_idx), ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white', zorder=6)

            # Plot target positions
            for agent_idx, target in enumerate(self.target_positions):
                ax.scatter(target[0], target[1], s=300, c=[self.colors[agent_idx]],
                          marker='*', edgecolors='black', linewidths=2, zorder=4, alpha=0.6)

            # Highlight collisions
            if frame in self.collision_steps:
                ax.text(self.grid_size / 2, -1, 'COLLISION!', ha='center',
                       fontsize=14, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            return []

        anim = FuncAnimation(fig, update, init_func=init, frames=max_steps,
                           interval=1000/fps, blit=True)

        # Save animation
        print(f"Saving animation to {save_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved successfully!")

        if show_final:
            # Show final frame
            self.plot_step(max_steps - 1, show=True)

        plt.close()

    def plot_summary(self, save_path=None, show=True):
        """
        Create a summary plot showing initial, middle, and final states.

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        if not self.trajectories[0]:
            print("No trajectory data to plot")
            return

        max_steps = len(self.trajectories[0])
        steps_to_plot = [0, max_steps // 2, max_steps - 1]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = ['Initial State', 'Mid-Training', 'Final State']

        for ax, step, title in zip(axes, steps_to_plot, titles):
            # Set up grid
            ax.set_xlim(-2, self.grid_size + 2)
            ax.set_ylim(-2, self.grid_size + 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'{title} (Step {step})', fontweight='bold')

            # Draw grid boundaries
            boundary = patches.Rectangle((0, 0), self.grid_size, self.grid_size,
                                         linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(boundary)

            # Plot trajectories
            for agent_idx in range(self.n_agents):
                traj = np.array(self.trajectories[agent_idx][:step+1])
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=self.colors[agent_idx],
                           alpha=0.3, linewidth=1)

            # Plot agent positions
            for agent_idx in range(self.n_agents):
                pos = self.trajectories[agent_idx][step]
                ax.scatter(pos[0], pos[1], s=200, c=[self.colors[agent_idx]],
                          edgecolors='black', linewidths=2, marker='o', zorder=5)
                ax.text(pos[0], pos[1], str(agent_idx), ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white', zorder=6)

            # Plot targets
            for agent_idx, target in enumerate(self.target_positions):
                ax.scatter(target[0], target[1], s=200, c=[self.colors[agent_idx]],
                          marker='*', edgecolors='black', linewidths=2, zorder=4, alpha=0.6)

        plt.suptitle('Multi-Agent Formation Control Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig


def visualize_from_env(env, actor, target_coords, n_steps=500, device='cpu',
                       save_dir='visualizations', create_animation=True):
    """
    Run the environment with a trained actor and visualize the results.

    Args:
        env: Grid environment
        actor: Trained actor network
        target_coords: Target coordinates
        n_steps: Maximum steps to run
        device: Device for inference
        save_dir: Directory to save visualizations
        create_animation: Whether to create an animated GIF
    """
    from .model import dict_obs_to_tensor

    os.makedirs(save_dir, exist_ok=True)

    # Initialize visualizer
    n_agents = env.n_agents
    vis = GridVisualizer(grid_size=env.grid_size, n_agents=n_agents)

    # Reset environment
    obs, info = env.reset(seed=42, target_coords=target_coords)

    # Collect trajectory
    step = 0
    while env.agents and step < n_steps:
        # Get current positions
        agent_positions = [env.agent_positions[f"agent_{i}"] for i in range(n_agents)]

        # Check for collisions
        collisions = []
        for i, agent in enumerate(env.agents):
            if agent in info and info.get(agent, {}).get('collision', False):
                collisions.append(i)

        # Add to trajectory
        vis.add_step(agent_positions, target_coords, collisions)

        # Get actions
        actions = {}
        for agent in env.agents:
            obs_tensor = dict_obs_to_tensor(obs[agent], device)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor, deterministic=True)
            actions[agent] = action.item()

        # Step
        obs, rewards, terminations, truncations, info = env.step(actions)
        step += 1

        # Check if done
        if all(terminations.get(agent, False) for agent in env.possible_agents):
            break

    # Create visualizations
    print(f"\nCreating visualizations in {save_dir}/...")

    # Summary plot
    summary_path = os.path.join(save_dir, 'formation_summary.png')
    vis.plot_summary(save_path=summary_path, show=False)
    print(f"✓ Summary plot saved: {summary_path}")

    # Final step plot
    final_path = os.path.join(save_dir, 'final_formation.png')
    vis.plot_step(step - 1, save_path=final_path, show=False)
    print(f"✓ Final formation saved: {final_path}")

    # Animation
    if create_animation:
        anim_path = os.path.join(save_dir, 'formation_animation.gif')
        vis.create_animation(save_path=anim_path, fps=10, show_final=False)
        print(f"✓ Animation saved: {anim_path}")

    print(f"\nVisualization complete! Files saved in {save_dir}/")

    return vis

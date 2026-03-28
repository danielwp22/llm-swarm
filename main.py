import torch
import argparse
from environment.grid_env import parallel_env
from environment.train import train_mappo, load_models
from environment.model import dict_obs_to_tensor
from llm.shape_gen import gen_shape


def run_trained_policy(env, actor, n_agents, target_coords, max_steps=500, device='cpu', render=True):
    """
    Run the trained policy on the environment.

    Args:
        env: Environment instance
        actor: Trained actor network
        n_agents: Number of agents
        target_coords: Target coordinates for formation
        max_steps: Maximum steps to run
        device: Device to run on
        render: Whether to render the environment
    """
    obs, info = env.reset(seed=42, target_coords=target_coords)

    total_reward = 0
    step = 0

    print(f"\n{'='*60}")
    print("Running trained policy...")
    print(f"{'='*60}\n")

    while env.agents and step < max_steps:
        actions = {}

        # Get actions from trained policy
        for agent in env.agents:
            obs_tensor = dict_obs_to_tensor(obs[agent], device)

            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor, deterministic=True)

            actions[agent] = action.item()

        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent in env.agents:
            if agent in rewards:
                total_reward += rewards[agent]

        step += 1

        # Check if episode ended
        if all(terminations.get(agent, False) for agent in env.possible_agents):
            print(f"\nSuccess! All agents reached their targets in {step} steps!")
            break

    if step >= max_steps:
        print(f"\nReached maximum steps ({max_steps})")

    avg_reward = total_reward / (n_agents * step) if step > 0 else 0
    print(f"\nTotal steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per agent per step: {avg_reward:.4f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Formation Control with CTDE')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'demo'],
                        help='Mode: train, eval, or demo')
    parser.add_argument('--n_agents', type=int, default=4,
                        help='Number of agents')
    parser.add_argument('--shape', type=str, default='circle',
                        help='Shape to form (e.g., circle, square, line, triangle)')
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--obs_radius', type=int, default=5,
                        help='Observation radius for agents')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--actor_path', type=str, default='models/actor_final.pt',
                        help='Path to trained actor model')
    parser.add_argument('--critic_path', type=str, default='models/critic_final.pt',
                        help='Path to trained critic model')
    parser.add_argument('--no_llm', action='store_true',
                        help='Skip LLM and use default circle formation')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Multi-Agent Formation Control with CTDE")
    print(f"{'='*60}\n")

    # Step 1: Generate target coordinates using LLM
    print(f"Step 1: Generating target coordinates for '{args.shape}'...")

    if args.no_llm:
        # Use default circle formation
        from llm.shape_gen import generate_default_circle
        target_coords = generate_default_circle(args.n_agents, grid_size=64)
        print(f"Using default circle formation (no LLM)")
    else:
        try:
            target_coords = gen_shape(args.shape, n_agents=args.n_agents, grid_size=64)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            print("Falling back to default circle formation")
            from llm.shape_gen import generate_default_circle
            target_coords = generate_default_circle(args.n_agents, grid_size=64)

    print(f"\nTarget coordinates: {target_coords}\n")

    # Step 2: Create environment
    print(f"Step 2: Creating environment...")
    env = parallel_env(
        render_mode="human" if args.mode in ['eval', 'demo'] else None,
        n_agents=args.n_agents,
        obs_radius=args.obs_radius
    )
    print(f"Environment created with {args.n_agents} agents\n")

    if args.mode == 'train':
        # Step 3: Train the policy
        print(f"Step 3: Training MAPPO policy...")
        print(f"Training for {args.n_episodes} episodes...\n")

        actor, critic = train_mappo(
            env=env,
            n_agents=args.n_agents,
            target_coords=target_coords,
            n_episodes=args.n_episodes,
            obs_radius=args.obs_radius,
            device=args.device,
            save_dir='models',
            log_interval=10,
        )

        print(f"\nStep 4: Running trained policy...")
        run_trained_policy(env, actor, args.n_agents, target_coords, device=args.device)

    elif args.mode == 'eval':
        # Load and evaluate trained model
        print(f"Step 3: Loading trained models...")
        try:
            actor, critic = load_models(
                args.actor_path,
                args.critic_path,
                n_agents=args.n_agents,
                obs_radius=args.obs_radius,
                device=args.device
            )
            print(f"Models loaded successfully\n")

            print(f"Step 4: Evaluating policy...")
            run_trained_policy(env, actor, args.n_agents, target_coords, device=args.device)

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train the model first using --mode train")

    elif args.mode == 'demo':
        # Run random actions for demonstration
        print(f"Step 3: Running random policy (demo mode)...")
        obs, info = env.reset(seed=42, target_coords=target_coords)

        step = 0
        max_steps = 500

        while env.agents and step < max_steps:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step += 1

            if all(terminations.get(agent, False) for agent in env.possible_agents):
                print(f"Episode ended at step {step}")
                break

        print(f"Demo completed in {step} steps")
        env.close()

    print(f"\n{'='*60}")
    print("Execution completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

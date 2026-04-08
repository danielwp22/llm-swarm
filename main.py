import torch
import argparse
import random
import numpy as np
from environment.grid_env import parallel_env
from environment.train import train_mappo, load_actor
from environment.train_improved import train_mappo_improved
from environment.model import dict_obs_to_tensor
from llm.shape_gen import gen_shape, generate_builtin_shape


def run_trained_policy(
    env,
    actor,
    n_agents,
    target_coords,
    max_steps=500,
    device='cpu',
    render=True,
    deterministic=True,
):
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
        deterministic: If True use argmax actions; if False sample stochastically
    """
    obs, info = env.reset(seed=42, target_coords=target_coords)

    total_reward = 0
    total_collisions = 0
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
                action, _ = actor.get_action(obs_tensor, deterministic=deterministic)

            actions[agent] = action.item()

        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards and track collisions
        for agent in env.agents:
            if agent in rewards:
                total_reward += rewards[agent]
            if agent in infos and infos[agent].get('collision', False):
                total_collisions += 1

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
    print(f"Total collisions: {total_collisions}")
    print(f"Average reward per agent per step: {avg_reward:.4f}")
    print(f"Collision rate: {total_collisions / step:.2f} collisions/step")

    env.close()


def _parse_shape_list(shape_list_arg):
    if not shape_list_arg:
        return []
    return [shape.strip() for shape in shape_list_arg.split(",") if shape.strip()]


def _resolve_target_coords(shape, n_agents, no_llm):
    builtin = generate_builtin_shape(shape, n_agents=n_agents, grid_size=64)
    if builtin is not None:
        return builtin

    if no_llm:
        raise ValueError(f"Shape '{shape}' is not a supported built-in shape and --no_llm was set.")

    return gen_shape(shape, n_agents=n_agents, grid_size=64)


def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Formation Control with CTDE')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'demo'],
                        help='Mode: train, eval, or demo')
    parser.add_argument('--trainer', type=str, default='baseline', choices=['baseline', 'improved'],
                        help='Training implementation to use for train/eval artifact paths')
    parser.add_argument('--n_agents', type=int, default=4,
                        help='Number of agents')
    parser.add_argument('--shape', type=str, default='circle',
                        help='Shape to form (e.g., circle, square, line, triangle)')
    parser.add_argument('--train_shapes', type=str, default=None,
                        help='Comma-separated list of shapes to sample during training, e.g. circle,triangle,square')
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--obs_radius', type=int, default=5,
                        help='Observation radius for agents')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto for automatic detection)')
    parser.add_argument('--actor_path', type=str, default=None,
                        help='Path to trained actor model (default: models/actor_{actor_type}_final.pt)')
    parser.add_argument('--critic_path', type=str, default=None,
                        help='Path to trained critic model (default: models/critic_{actor_type}_final.pt)')
    parser.add_argument('--no_llm', action='store_true',
                        help='Skip LLM and use default circle formation')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations (plots and animations)')
    parser.add_argument('--vis_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--no_animation', action='store_true',
                        help='Skip animation generation (faster)')
    parser.add_argument('--actor_type', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='Actor architecture type: cnn (default, convolutional) or mlp (simpler)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Initial entropy coefficient for PPO training')
    parser.add_argument('--entropy_coef_end', type=float, default=0.001,
                        help='Final entropy coefficient for linear decay during training')
    parser.add_argument('--num_mini_batch', type=int, default=4,
                        help='Number of PPO mini-batches per epoch')
    parser.add_argument('--ppo_epochs', type=int, default=None,
                        help='Override PPO epochs. Defaults: baseline=4, improved=15')
    parser.add_argument('--critic_input_type', type=str, default='agent_specific',
                        choices=['shared', 'agent_specific', 'full_local_concat'],
                        help='Improved trainer critic input: compact shared state, paper-style agent-specific state, or full local-grid concat ablation')
    parser.add_argument('--no_clipped_value_loss', action='store_true',
                        help='Disable clipped value loss for critic update')
    parser.add_argument('--stochastic_eval', action='store_true',
                        help='In eval, sample actions stochastically instead of argmax')
    parser.add_argument('--easy_curriculum', action='store_true',
                        help='Training helper: force n_agents=4, no_llm, shape=circle')
    parser.add_argument('--random_targets', action='store_true',
                        help='Train with random target coordinates sampled each episode (generalizes across any formation)')
    parser.add_argument('--min_target_distance', type=int, default=0,
                        help='Minimum Chebyshev distance between random target pairs (0 = auto)')
    parser.add_argument('--llm_agent_count', action='store_true',
                        help='Let LLM decide n_agents based on the shape (eval/demo only; actor is n_agents-independent)')
    parser.add_argument('--min_agents', type=int, default=2,
                        help='Minimum agents when --llm_agent_count is used')
    parser.add_argument('--max_agents', type=int, default=16,
                        help='Maximum agents when --llm_agent_count is used')

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.mode == 'train' and args.easy_curriculum:
        args.n_agents = 4
        args.no_llm = True
        args.shape = 'circle'
        print("Easy curriculum enabled: using n_agents=4, no_llm=True, shape='circle'")

    print(f"\n{'='*60}")
    print("Multi-Agent Formation Control with CTDE")
    print(f"{'='*60}")
    print(f"Device: {args.device.upper()}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Actor Type: {args.actor_type.upper()}")
    print(f"Trainer: {args.trainer.upper()}")
    if args.trainer == 'improved':
        print(f"Critic Input: {args.critic_input_type.upper()}")
    print(f"{'='*60}\n")

    train_shapes = _parse_shape_list(args.train_shapes) if args.mode == 'train' else []

    # LLM agent count: resolve n_agents before Step 1 and env creation (eval/demo only)
    _skip_step1 = False
    if args.llm_agent_count:
        if args.mode == 'train':
            print("Warning: --llm_agent_count is only effective in eval/demo mode. Ignoring.")
        else:
            from llm.shape_gen import BUILTIN_NATURAL_COUNTS, get_completion_with_agent_count
            builtin_n = BUILTIN_NATURAL_COUNTS.get((args.shape or "").strip().lower())
            if builtin_n is not None:
                args.n_agents = builtin_n
                target_coords = generate_builtin_shape(args.shape, n_agents=builtin_n, grid_size=64)
            elif args.no_llm:
                raise ValueError("--llm_agent_count with --no_llm requires a built-in shape (circle, square, triangle, line).")
            else:
                args.n_agents, target_coords = get_completion_with_agent_count(
                    args.shape, grid_size=64, min_agents=args.min_agents, max_agents=args.max_agents)
            target_coords_sampler = None
            print(f"LLM chose {args.n_agents} agents for '{args.shape}'")
            print(f"Target coordinates: {target_coords}\n")
            _skip_step1 = True

    # Step 1: Generate target coordinates
    if _skip_step1:
        pass
    elif args.mode == 'train' and args.random_targets:
        from llm.shape_gen import generate_random_targets
        _rng = np.random.default_rng(42)
        _min_dist = args.min_target_distance if args.min_target_distance > 0 else max(2, 64 // (args.n_agents + 2))
        target_coords = generate_random_targets(args.n_agents, min_distance=_min_dist, rng=_rng)
        target_coords_sampler = lambda ep: generate_random_targets(args.n_agents, min_distance=_min_dist, rng=_rng)
        print(f"Step 1: Random target training enabled (min_distance={_min_dist})")
        print(f"Example target coordinates: {target_coords}\n")
    elif train_shapes:
        print(f"Step 1: Precomputing training target coordinates for shapes: {train_shapes}...")
        target_bank = {}
        for shape_name in train_shapes:
            try:
                target_bank[shape_name] = _resolve_target_coords(shape_name, args.n_agents, args.no_llm)
            except Exception as e:
                print(f"Failed to generate shape '{shape_name}': {e}")
                raise

        sampler_rng = random.Random(42)
        sampled_shapes = train_shapes[:]

        def target_coords_sampler(_episode):
            shape_name = sampler_rng.choice(sampled_shapes)
            return target_bank[shape_name]

        eval_shape = args.shape if args.shape in target_bank else train_shapes[0]
        target_coords = target_bank[eval_shape]
        print(f"Training shapes ready. Post-train eval shape: '{eval_shape}'")
        print(f"\nExample target coordinates ({eval_shape}): {target_coords}\n")
    else:
        print(f"Step 1: Generating target coordinates for '{args.shape}'...")
        try:
            target_coords = _resolve_target_coords(args.shape, args.n_agents, args.no_llm)
        except Exception as e:
            print(f"Target generation failed: {e}")
            print("Falling back to built-in circle formation")
            target_coords = generate_builtin_shape("circle", args.n_agents, 64)
        target_coords_sampler = None
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
        ppo_epochs = args.ppo_epochs
        if ppo_epochs is None:
            ppo_epochs = 15 if args.trainer == 'improved' else 4

        save_dir = f'models/{args.trainer}'

        # Step 3: Train the policy
        print(f"Step 3: Training {args.trainer} MAPPO policy...")
        print(f"Training for {args.n_episodes} episodes...\n")

        if args.trainer == 'improved':
            actor, critic, history = train_mappo_improved(
                env=env,
                n_agents=args.n_agents,
                target_coords=target_coords,
                target_coords_sampler=target_coords_sampler,
                n_episodes=args.n_episodes,
                obs_radius=args.obs_radius,
                device=args.device,
                save_dir=save_dir,
                log_interval=10,
                actor_type=args.actor_type,
                entropy_coef=args.entropy_coef,
                entropy_coef_end=args.entropy_coef_end,
                num_mini_batch=args.num_mini_batch,
                ppo_epochs=ppo_epochs,
                critic_input_type=args.critic_input_type,
                use_clipped_value_loss=not args.no_clipped_value_loss,
            )
        else:
            actor, critic, history = train_mappo(
                env=env,
                n_agents=args.n_agents,
                target_coords=target_coords,
                target_coords_sampler=target_coords_sampler,
                n_episodes=args.n_episodes,
                obs_radius=args.obs_radius,
                device=args.device,
                save_dir=save_dir,
                log_interval=10,
                actor_type=args.actor_type,
                entropy_coef=args.entropy_coef,
                entropy_coef_end=args.entropy_coef_end,
                num_mini_batch=args.num_mini_batch,
                ppo_epochs=ppo_epochs,
                use_clipped_value_loss=not args.no_clipped_value_loss,
            )

        print(f"\nStep 4: Running trained policy...")
        run_trained_policy(env, actor, args.n_agents, target_coords, device=args.device, deterministic=True)

        # Visualization
        if args.visualize:
            print(f"\nStep 5: Creating visualizations...")

            # Plot training metrics
            from environment.visualize import plot_training_metrics
            print(f"\nGenerating training metrics plots...")
            plot_training_metrics(history, save_dir=args.vis_dir, show=False)

            # Plot formation visualization
            from environment.visualize import visualize_from_env
            print(f"\nGenerating formation visualizations...")
            visualize_from_env(
                env=parallel_env(n_agents=args.n_agents, obs_radius=args.obs_radius),
                actor=actor,
                target_coords=target_coords,
                device=args.device,
                save_dir=args.vis_dir,
                create_animation=not args.no_animation
            )

    elif args.mode == 'eval':
        # Load and evaluate trained model
        print(f"Step 3: Loading trained actor...")

        # Use architecture-specific default paths if not provided
        actor_path = args.actor_path if args.actor_path else f'models/{args.trainer}/actor_{args.actor_type}_final.pt'
        if args.critic_path:
            print("Note: --critic_path is ignored in eval mode (CTDE execution only needs the actor).")

        try:
            actor = load_actor(
                actor_path,
                obs_radius=args.obs_radius,
                device=args.device,
                actor_type=args.actor_type,
            )
            print(f"Actor loaded successfully\n")

            print(f"Step 4: Evaluating policy...")
            run_trained_policy(
                env,
                actor,
                args.n_agents,
                target_coords,
                device=args.device,
                deterministic=not args.stochastic_eval,
            )

            # Visualization
            if args.visualize:
                print(f"\nStep 5: Creating visualizations...")
                from environment.visualize import visualize_from_env
                visualize_from_env(
                    env=parallel_env(n_agents=args.n_agents, obs_radius=args.obs_radius),
                    actor=actor,
                    target_coords=target_coords,
                    device=args.device,
                    save_dir=args.vis_dir,
                    create_animation=not args.no_animation,
                    deterministic=not args.stochastic_eval,
                )

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Check that --n_agents, --obs_radius, and --actor_type match the training run.")

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

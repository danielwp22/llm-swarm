import argparse
import heapq
import os

import numpy as np

from environment.grid_env import ACTIONS, parallel_env
from environment.visualize import GridVisualizer
from llm.shape_gen import gen_shape, generate_builtin_shape


ACTION_TO_DELTA = ACTIONS
DELTA_TO_ACTION = {tuple(delta): action for action, delta in ACTIONS.items()}


def resolve_target_coords(shape, n_agents, no_llm):
    builtin = generate_builtin_shape(shape, n_agents=n_agents, grid_size=64)
    if builtin is not None:
        return builtin
    if no_llm:
        raise ValueError(f"Shape '{shape}' is not a built-in shape and --no_llm was set.")
    return gen_shape(shape, n_agents=n_agents, grid_size=64)


def manhattan_chebyshev_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)


def violates_constraint(agent, curr, nxt, time_step, constraints):
    for constraint in constraints:
        if constraint["agent"] != agent:
            continue
        if constraint["time"] != time_step:
            continue
        if constraint["type"] == "vertex" and tuple(constraint["loc"]) == tuple(nxt):
            return True
        if constraint["type"] == "edge" and tuple(constraint["from"]) == tuple(curr) and tuple(constraint["to"]) == tuple(nxt):
            return True
    return False


def future_goal_blocked(agent, goal, time_step, constraints):
    for constraint in constraints:
        if constraint["agent"] != agent or constraint["type"] != "vertex":
            continue
        if constraint["time"] >= time_step and tuple(constraint["loc"]) == tuple(goal):
            return True
    return False


def low_level_a_star(agent, start, goal, grid_size, constraints):
    max_constraint_time = 0
    for constraint in constraints:
        if constraint["agent"] == agent:
            max_constraint_time = max(max_constraint_time, constraint["time"])

    start = tuple(start)
    goal = tuple(goal)
    frontier = []
    heapq.heappush(frontier, (manhattan_chebyshev_distance(start, goal), 0, start, 0))
    came_from = {(start, 0): None}
    g_score = {(start, 0): 0}

    while frontier:
        _, cost, current, time_step = heapq.heappop(frontier)

        if current == goal and time_step >= max_constraint_time and not future_goal_blocked(agent, goal, time_step, constraints):
            path = []
            node = (current, time_step)
            while node is not None:
                path.append(node[0])
                node = came_from[node]
            return list(reversed(path))

        next_time = time_step + 1
        for delta in ACTION_TO_DELTA.values():
            nxt = (
                min(grid_size - 1, max(0, current[0] + delta[0])),
                min(grid_size - 1, max(0, current[1] + delta[1])),
            )
            if violates_constraint(agent, current, nxt, next_time, constraints):
                continue

            next_node = (nxt, next_time)
            tentative_g = cost + 1
            if tentative_g < g_score.get(next_node, float("inf")):
                g_score[next_node] = tentative_g
                came_from[next_node] = (current, time_step)
                f_score = tentative_g + manhattan_chebyshev_distance(nxt, goal)
                heapq.heappush(frontier, (f_score, tentative_g, nxt, next_time))

    return None


def pad_path(path, length):
    if len(path) >= length:
        return path
    return path + [path[-1]] * (length - len(path))


def detect_conflict(paths):
    makespan = max(len(path) for path in paths.values())
    padded = {agent: pad_path(path, makespan) for agent, path in paths.items()}

    for t in range(makespan):
        agents = list(padded.keys())
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                pos_a = padded[agent_a][t]
                pos_b = padded[agent_b][t]
                if pos_a == pos_b:
                    return {
                        "time": t,
                        "type": "vertex",
                        "agents": (agent_a, agent_b),
                        "loc": pos_a,
                    }
                if t > 0:
                    prev_a = padded[agent_a][t - 1]
                    prev_b = padded[agent_b][t - 1]
                    if prev_a == pos_b and prev_b == pos_a:
                        return {
                            "time": t,
                            "type": "edge",
                            "agents": (agent_a, agent_b),
                            "from_to": ((prev_a, pos_a), (prev_b, pos_b)),
                        }
    return None


def compute_cost(paths):
    return sum(len(path) - 1 for path in paths.values())


def cbs_solve(starts, goals, grid_size):
    root_constraints = []
    root_paths = {}
    for agent, start in starts.items():
        path = low_level_a_star(agent, start, goals[agent], grid_size, root_constraints)
        if path is None:
            return None
        root_paths[agent] = path

    open_list = []
    node_id = 0
    root_conflict = detect_conflict(root_paths)
    heapq.heappush(open_list, (compute_cost(root_paths), 0 if root_conflict is None else 1, node_id, root_constraints, root_paths))
    node_id += 1

    while open_list:
        _, _, _, constraints, paths = heapq.heappop(open_list)
        conflict = detect_conflict(paths)
        if conflict is None:
            return paths

        for branch_idx, agent in enumerate(conflict["agents"]):
            new_constraints = list(constraints)
            if conflict["type"] == "vertex":
                new_constraints.append({
                    "agent": agent,
                    "type": "vertex",
                    "loc": conflict["loc"],
                    "time": conflict["time"],
                })
            else:
                edge = conflict["from_to"][branch_idx]
                new_constraints.append({
                    "agent": agent,
                    "type": "edge",
                    "from": edge[0],
                    "to": edge[1],
                    "time": conflict["time"],
                })

            new_paths = dict(paths)
            replanned = low_level_a_star(agent, starts[agent], goals[agent], grid_size, new_constraints)
            if replanned is None:
                continue
            new_paths[agent] = replanned
            new_conflict = detect_conflict(new_paths)
            heapq.heappush(
                open_list,
                (compute_cost(new_paths), 0 if new_conflict is None else 1, node_id, new_constraints, new_paths),
            )
            node_id += 1

    return None


def path_to_action(curr, nxt):
    delta = (int(nxt[0] - curr[0]), int(nxt[1] - curr[1]))
    return DELTA_TO_ACTION[delta]


def simulate_and_visualize(paths, target_coords, n_agents, obs_radius, vis_dir, no_animation):
    os.makedirs(vis_dir, exist_ok=True)
    env = parallel_env(n_agents=n_agents, obs_radius=obs_radius)
    obs, info = env.reset(seed=42, target_coords=target_coords)

    vis = GridVisualizer(grid_size=env.grid_size, n_agents=n_agents)
    ordered_agents = env.possible_agents
    makespan = max(len(path) for path in paths.values())

    for t in range(makespan):
        agent_positions = [env.agent_positions[f"agent_{i}"] for i in range(n_agents)]
        vis.add_step(agent_positions, target_coords, collisions=[])

        actions = {}
        for agent in ordered_agents:
            path = pad_path(paths[agent], makespan)
            curr = path[t]
            nxt = path[t + 1] if t + 1 < makespan else path[t]
            actions[agent] = path_to_action(curr, nxt)

        obs, rewards, terminations, truncations, info = env.step(actions)
        if not env.agents:
            break

    # Record the final settled state after the last action so the visualization
    # includes the actual goal-reaching configuration.
    final_positions = [env.agent_positions[f"agent_{i}"] for i in range(n_agents)]
    vis.add_step(final_positions, target_coords, collisions=[])

    summary_path = os.path.join(vis_dir, "cbs_summary.png")
    final_path = os.path.join(vis_dir, "cbs_final.png")
    vis.plot_summary(save_path=summary_path, show=False)
    vis.plot_step(len(vis.trajectories[0]) - 1, save_path=final_path, show=False)
    if not no_animation:
        anim_path = os.path.join(vis_dir, "cbs_animation.gif")
        vis.create_animation(save_path=anim_path, fps=10, show_final=False)

    return compute_cost(paths), makespan - 1


def main():
    parser = argparse.ArgumentParser(description="Conflict-Based Search baseline for formation planning")
    parser.add_argument("--shape", type=str, default="circle")
    parser.add_argument("--n_agents", type=int, default=8)
    parser.add_argument("--obs_radius", type=int, default=7)
    parser.add_argument("--vis_dir", type=str, default="visualizations/cbs")
    parser.add_argument("--no_animation", action="store_true")
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    target_coords = resolve_target_coords(args.shape, args.n_agents, args.no_llm)
    env = parallel_env(n_agents=args.n_agents, obs_radius=args.obs_radius)
    obs, info = env.reset(seed=42, target_coords=target_coords)

    starts = {agent: tuple(env.agent_positions[agent].astype(int)) for agent in env.possible_agents}
    goals = {agent: tuple(env.target_positions[agent].astype(int)) for agent in env.possible_agents}
    env.close()

    print(f"Starts: {starts}")
    print(f"Goals: {goals}")
    print("Running CBS...")
    paths = cbs_solve(starts, goals, grid_size=64)
    if paths is None:
        raise RuntimeError("CBS failed to find a conflict-free plan.")

    total_cost, makespan = simulate_and_visualize(
        paths,
        target_coords=target_coords,
        n_agents=args.n_agents,
        obs_radius=args.obs_radius,
        vis_dir=args.vis_dir,
        no_animation=args.no_animation,
    )
    print(f"CBS solved successfully. Sum of costs: {total_cost}, makespan: {makespan}")
    print(f"Visualization saved to {args.vis_dir}")


if __name__ == "__main__":
    main()

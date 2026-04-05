import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from llm.shape_gen import gen_shape, generate_builtin_shape


def resolve_target_coords(shape, n_agents, no_llm):
    builtin = generate_builtin_shape(shape, n_agents=n_agents, grid_size=64)
    if builtin is not None:
        return builtin
    if no_llm:
        raise ValueError(f"Shape '{shape}' is not a built-in shape and --no_llm was set.")
    return gen_shape(shape, n_agents=n_agents, grid_size=64)


def preview_shape(shape, coords, save_path=None, show=True, grid_size=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(coords)))

    ax.set_xlim(-2, grid_size + 2)
    ax.set_ylim(-2, grid_size + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f"Shape Preview: {shape}", fontweight='bold')

    boundary = patches.Rectangle((0, 0), grid_size, grid_size, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(boundary)

    coords_arr = np.array(coords)
    if len(coords_arr) > 1:
        closed = np.vstack([coords_arr, coords_arr[0]])
        ax.plot(closed[:, 0], closed[:, 1], '--', color='gray', alpha=0.5, linewidth=1)

    for i, coord in enumerate(coords):
        ax.scatter(coord[0], coord[1], s=260, c=[colors[i]], edgecolors='black', linewidths=2, marker='*', zorder=5)
        ax.text(coord[0], coord[1], str(i), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Preview generated target shapes without training")
    parser.add_argument("--shape", type=str, required=True, help="Shape description, e.g. circle, triangle, tree")
    parser.add_argument("--n_agents", type=int, default=8)
    parser.add_argument("--vis_dir", type=str, default="visualizations/shape_preview")
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    coords = resolve_target_coords(args.shape, args.n_agents, args.no_llm)
    os.makedirs(args.vis_dir, exist_ok=True)
    safe_shape = args.shape.replace(" ", "_")
    save_path = os.path.join(args.vis_dir, f"{safe_shape}_{args.n_agents}.png")

    print(f"Generated coordinates for '{args.shape}':")
    for i, coord in enumerate(coords):
        print(f"  Agent {i}: {coord}")

    preview_shape(args.shape, coords, save_path=save_path, show=not args.no_show)
    print(f"Preview saved to {save_path}")


if __name__ == "__main__":
    main()

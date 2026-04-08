import os
import json
import math
import numpy as np
from openai import OpenAI

# The client automatically picks up the OPENAI_API_KEY environment variable
# Alternatively, you can pass the key explicitly: client = OpenAI(api_key="YOUR_API_KEY")
client = OpenAI()

DEFAULT_SHAPE_MODEL = os.getenv("OPENAI_SHAPE_MODEL", "gpt-5.2")


def _normalize_shape_name(prompt):
    return (prompt or "").strip().lower()


def generate_default_line(n_agents, grid_size=64):
    center = grid_size // 2
    start_x = max(2, center - n_agents // 2)
    y = center
    return [[min(grid_size - 3, start_x + i), y] for i in range(n_agents)]


def generate_default_square(n_agents, grid_size=64):
    center = grid_size // 2
    side = max(8, grid_size // 3)
    perimeter = max(4, 4 * side)
    coordinates = []
    for i in range(n_agents):
        t = int(round(i * perimeter / n_agents)) % perimeter
        if t < side:
            x = center - side // 2 + t
            y = center - side // 2
        elif t < 2 * side:
            x = center + side // 2
            y = center - side // 2 + (t - side)
        elif t < 3 * side:
            x = center + side // 2 - (t - 2 * side)
            y = center + side // 2
        else:
            x = center - side // 2
            y = center + side // 2 - (t - 3 * side)
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))
        coordinates.append([x, y])
    return coordinates


def generate_default_triangle(n_agents, grid_size=64):
    center = grid_size // 2
    radius = grid_size // 3
    vertices = np.array([
        [center, center + radius],
        [center - radius, center - radius],
        [center + radius, center - radius],
    ], dtype=np.float32)
    # Distribute n_agents evenly along the full perimeter.
    # Each edge has equal length (equilateral), so space agents uniformly by fraction.
    coordinates = []
    for i in range(n_agents):
        t = i / n_agents  # fraction along total perimeter [0, 1)
        t_edge = t * 3    # scaled to [0, 3)
        edge_idx = int(t_edge) % 3
        edge_progress = t_edge - int(t_edge)
        start = vertices[edge_idx]
        end = vertices[(edge_idx + 1) % 3]
        point = start + edge_progress * (end - start)
        x = int(round(point[0]))
        y = int(round(point[1]))
        coordinates.append([max(0, min(x, grid_size - 1)), max(0, min(y, grid_size - 1))])
    return coordinates


def generate_random_targets(n_agents, grid_size=64, min_distance=0, rng=None):
    """
    Generate random target coordinates with optional minimum Chebyshev distance between pairs.

    Args:
        n_agents: Number of target positions to generate
        grid_size: Grid size (coordinates in [0, grid_size-1])
        min_distance: Minimum Chebyshev distance between any two targets (0 = pure uniform)
        rng: Optional numpy.random.Generator for reproducibility
    Returns:
        List of [x, y] integer coordinate pairs
    """
    if rng is None:
        rng = np.random.default_rng()
    coords = []
    for _ in range(n_agents):
        for _ in range(1000):
            x = int(rng.integers(0, grid_size))
            y = int(rng.integers(0, grid_size))
            if all(max(abs(x - cx), abs(y - cy)) >= min_distance for cx, cy in coords):
                coords.append([x, y])
                break
        else:
            coords.append([int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size))])
    return coords


# Natural agent counts for built-in shapes (used when LLM decides n_agents)
BUILTIN_NATURAL_COUNTS = {"circle": 8, "square": 8, "triangle": 3, "line": 4}


def get_completion_with_agent_count(prompt, grid_size=64, min_agents=2, max_agents=16):
    """
    Like get_completion but asks the LLM to decide the number of agents.

    Args:
        prompt: Natural language shape description
        grid_size: Grid size
        min_agents: Minimum agents the LLM may choose
        max_agents: Maximum agents the LLM may choose
    Returns:
        (n_agents: int, coordinates: List[[x, y]])
    """
    system_prompt = f"""You are placing agents on a {grid_size}x{grid_size} grid to form a recognizable shape.

Coordinate system: x increases RIGHT, y increases UP. So y=0 is the BOTTOM, y={grid_size-1} is the TOP.
Center of the grid is ({grid_size//2}, {grid_size//2}).


Rules:
- Draw the object in a recognizable way using the dots. 
- A viewer seeing only the dot positions should immediately recognize the shape
- For shapes with distinct top/bottom features (faces, animals): high y = top, low y = bottom
- Return ONLY valid JSON, no extra text

Format: {{"n_agents": <int>, "coordinates": [[x1, y1], [x2, y2], ...]}}

Example — "triangle" (3 agents, one per vertex, tip pointing up):
{{"n_agents": 3, "coordinates": [[32, 54], [10, 10], [54, 10]]}}

IMPORTANT:
- n_agents must be an integer between {min_agents} and {max_agents}
- All coordinates must be integers in [0, {grid_size-1}]
- coordinates array must have exactly n_agents entries
- Output must be valid JSON (double quotes, proper brackets)"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_SHAPE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a {prompt} formation"}
            ],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content.strip())
        n = max(min_agents, min(max_agents, int(result["n_agents"])))
        coords = [
            [max(0, min(int(c[0]), grid_size - 1)), max(0, min(int(c[1]), grid_size - 1))]
            for c in result["coordinates"][:n]
        ]
        while len(coords) < n:
            coords.append(generate_default_circle(1, grid_size)[0])
        return n, coords
    except Exception as e:
        print(f"LLM agent-count query failed: {e}. Falling back to 4-agent circle.")
        return 4, generate_default_circle(4, grid_size)


def generate_builtin_shape(prompt, n_agents=4, grid_size=64):
    shape = _normalize_shape_name(prompt)
    if shape == "circle":
        return generate_default_circle(n_agents, grid_size)
    if shape == "line":
        return generate_default_line(n_agents, grid_size)
    if shape == "square":
        return generate_default_square(n_agents, grid_size)
    if shape == "triangle":
        return generate_default_triangle(n_agents, grid_size)
    return None

def get_completion(prompt, n_agents=4, grid_size=64):
    """
    Sends a prompt to the OpenAI API and returns the response content.

    Args:
        prompt: Natural language description of desired shape
        n_agents: Number of agents/coordinate points to generate
        grid_size: Size of the grid (default 64x64)

    Returns:
        List of coordinates [[x1, y1], [x2, y2], ...]
    """
    system_prompt = f"""You are an intelligent coordinates generator for multi-agent formation control on a {grid_size}x{grid_size} grid.

Your task:
1. Given a shape description and number of agents, generate {n_agents} coordinates that form the shape
2. Coordinates must be within bounds: 0 to {grid_size-1} for both x and y
3. Spread points evenly to outline the shape (not fill it)
4. Return ONLY a valid JSON array of coordinates, no extra text

Format: [[x1, y1], [x2, y2], ..., [xN, yN]]

Example for "circle with 8 agents":
[[32, 48], [45, 45], [48, 32], [45, 19], [32, 16], [19, 19], [16, 32], [19, 45]]

IMPORTANT:
- All coordinates must be integers between 0 and {grid_size-1}
- Return exactly {n_agents} coordinate pairs
- Output must be valid JSON (use double quotes, proper brackets)
- Do not add any explanations or extra text"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_SHAPE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a {prompt} with {n_agents} agents"}
            ],
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # Parse the JSON response
        try:
            coordinates = json.loads(content)

            # Validate coordinates
            if not isinstance(coordinates, list):
                raise ValueError("Response is not a list")

            if len(coordinates) != n_agents:
                print(f"Warning: Expected {n_agents} coordinates, got {len(coordinates)}")

            # Ensure coordinates are within bounds and convert to integers
            validated_coords = []
            for coord in coordinates:
                if len(coord) != 2:
                    raise ValueError(f"Invalid coordinate: {coord}")
                x = int(coord[0])
                y = int(coord[1])
                # Clip to grid bounds
                x = max(0, min(x, grid_size - 1))
                y = max(0, min(y, grid_size - 1))
                validated_coords.append([x, y])

            return validated_coords

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {content}")
            print(f"Error: {e}")
            # Fallback: return evenly spaced points in a circle
            return generate_default_circle(n_agents, grid_size)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Fallback: return evenly spaced points in a circle
        return generate_default_circle(n_agents, grid_size)


def generate_default_circle(n_agents, grid_size=64):
    """
    Generate a default circular formation when LLM fails.

    Args:
        n_agents: Number of agents
        grid_size: Size of the grid

    Returns:
        List of coordinates forming a circle
    """
    center = grid_size // 2
    radius = grid_size // 3

    coordinates = []
    for i in range(n_agents):
        angle = 2 * math.pi * i / n_agents
        x = int(center + radius * math.cos(angle))
        y = int(center + radius * math.sin(angle))
        # Clip to bounds
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))
        coordinates.append([x, y])

    return coordinates


def gen_shape(prompt=None, n_agents=4, grid_size=64):
    """
    Main function to generate shape coordinates.

    Args:
        prompt: Shape description (if None, will prompt user for input)
        n_agents: Number of agents
        grid_size: Grid size

    Returns:
        List of coordinates
    """
    if prompt is None:
        print(f"Enter a shape description (e.g., 'circle', 'square', 'triangle', 'line'):")
        prompt = input("> ")

    print(f"\nGenerating {prompt} formation for {n_agents} agents on {grid_size}x{grid_size} grid...")

    builtin = generate_builtin_shape(prompt, n_agents, grid_size)
    if builtin is not None:
        coordinates = builtin
    else:
        coordinates = get_completion(prompt, n_agents, grid_size)

    print(f"\nGenerated coordinates:")
    for i, coord in enumerate(coordinates):
        print(f"  Agent {i}: {coord}")

    return coordinates


if __name__ == "__main__":
    # Test the shape generator
    test_shapes = ["circle", "square", "line", "triangle"]

    for shape in test_shapes:
        print(f"\n{'='*50}")
        coords = gen_shape(shape, n_agents=8, grid_size=64)
        print(f"Shape: {shape}")
        print(f"Coordinates: {coords}")

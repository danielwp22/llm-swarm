import os
import json
import math
import numpy as np
from openai import OpenAI

# The client automatically picks up the OPENAI_API_KEY environment variable
# Alternatively, you can pass the key explicitly: client = OpenAI(api_key="YOUR_API_KEY")
client = OpenAI()

DEFAULT_SHAPE_MODEL = os.getenv("OPENAI_SHAPE_MODEL", "gpt-4o")


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


def generate_default_diamond(n_agents, grid_size=64):
    center = grid_size // 2
    radius = grid_size // 3
    vertices = np.array([
        [center, center + radius],   # top
        [center + radius, center],   # right
        [center, center - radius],   # bottom
        [center - radius, center],   # left
    ], dtype=np.float32)
    coords = []
    for i in range(n_agents):
        t = i / n_agents * 4        # [0, 4)
        e = int(t) % 4
        p = t - int(t)
        pt = vertices[e] + p * (vertices[(e + 1) % 4] - vertices[e])
        coords.append([
            int(round(float(np.clip(pt[0], 0, grid_size - 1)))),
            int(round(float(np.clip(pt[1], 0, grid_size - 1)))),
        ])
    return coords


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


def _build_agent_count_prompt(variant, grid_size, min_agents, max_agents):
    """Return system prompt for get_completion_with_agent_count variants."""
    fmt = f'{{"n_agents": <int>, "coordinates": [[x1, y1], [x2, y2], ...]}}'
    ex = f'{{"n_agents": 3, "coordinates": [[32, 54], [10, 10], [54, 10]]}}'
    if variant == "minimal":
        return (
            f"Place between {min_agents} and {max_agents} agents on a {grid_size}x{grid_size} grid to form the requested shape. "
            f"Choose the number of agents that best represents the shape. "
            f"Return ONLY valid JSON: {fmt}"
        )
    if variant == "detailed":
        return (
            f"You are placing agents on a {grid_size}x{grid_size} grid to form a recognizable shape.\n\n"
            f"Coordinate system: x increases RIGHT, y increases UP. y=0 is BOTTOM, y={grid_size-1} is TOP.\n"
            f"Center of the grid is ({grid_size//2}, {grid_size//2}).\n\n"
            f"Rules:\n"
            f"- Draw the object in a recognizable way using the dots.\n"
            f"- A viewer seeing only the dot positions should immediately recognize the shape\n"
            f"- For shapes with distinct top/bottom features: high y = top, low y = bottom\n"
            f"- Use enough agents to capture the shape's key features\n"
            f"- Return ONLY valid JSON, no extra text\n\n"
            f"Format: {fmt}\n\n"
            f"Example — \"triangle\": {ex}\n"
            f"Example — \"star\" (10 agents, alternating outer/inner points):\n"
            f'{{"n_agents": 10, "coordinates": [[32,52],[39,39],[52,32],[39,25],[32,12],[25,25],[12,32],[25,39],[20,20],[44,20]]}}\n'
            f"Example — \"house\" (12 agents, square base + triangular roof):\n"
            f'{{"n_agents": 12, "coordinates": [[16,16],[32,16],[48,16],[48,24],[48,32],[32,32],[16,32],[16,24],[24,40],[32,50],[40,40],[32,32]]}}\n\n'
            f"IMPORTANT:\n"
            f"- n_agents must be an integer between {min_agents} and {max_agents}\n"
            f"- All coordinates must be integers in [0, {grid_size-1}]\n"
            f"- coordinates array must have exactly n_agents entries\n"
            f"- Output must be valid JSON (double quotes, proper brackets)"
        )
    # standard (default)
    return (
        f"You are placing agents on a {grid_size}x{grid_size} grid to form a recognizable shape.\n\n"
        f"Coordinate system: x increases RIGHT, y increases UP. So y=0 is the BOTTOM, y={grid_size-1} is the TOP.\n"
        f"Center of the grid is ({grid_size//2}, {grid_size//2}).\n\n\n"
        f"Rules:\n"
        f"- Draw the object in a recognizable way using the dots. \n"
        f"- A viewer seeing only the dot positions should immediately recognize the shape\n"
        f"- For shapes with distinct top/bottom features (faces, animals): high y = top, low y = bottom\n"
        f"- Return ONLY valid JSON, no extra text\n\n"
        f"Format: {fmt}\n\n"
        f"Example — \"triangle\" (3 agents, one per vertex, tip pointing up):\n"
        f"{ex}\n\n"
        f"IMPORTANT:\n"
        f"- n_agents must be an integer between {min_agents} and {max_agents}\n"
        f"- All coordinates must be integers in [0, {grid_size-1}]\n"
        f"- coordinates array must have exactly n_agents entries\n"
        f"- Output must be valid JSON (double quotes, proper brackets)"
    )


def get_completion_with_agent_count(prompt, grid_size=64, min_agents=2, max_agents=16,
                                    model=None, prompt_variant="standard"):
    """
    Like get_completion but asks the LLM to decide the number of agents.

    Args:
        prompt: Natural language shape description
        grid_size: Grid size
        min_agents: Minimum agents the LLM may choose
        max_agents: Maximum agents the LLM may choose
        model: OpenAI model name override (default: DEFAULT_SHAPE_MODEL)
        prompt_variant: "standard" | "minimal" | "detailed"
    Returns:
        (n_agents: int, coordinates: List[[x, y]])
    """
    model = model or DEFAULT_SHAPE_MODEL
    system_prompt = _build_agent_count_prompt(prompt_variant, grid_size, min_agents, max_agents)

    try:
        response = client.chat.completions.create(
            model=model,
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
    if shape == "diamond":
        return generate_default_diamond(n_agents, grid_size)
    return None

def _build_system_prompt(variant, n_agents, grid_size):
    """Return the system prompt for the requested variant."""
    if variant == "minimal":
        return (
            f"Generate {n_agents} [x,y] integer coordinates on a {grid_size}x{grid_size} grid "
            f"that outline the requested shape. "
            f"Return ONLY a JSON array: [[x1,y1],[x2,y2],...]. No extra text."
        )
    if variant == "detailed":
        return (
            f"You are an intelligent coordinates generator for multi-agent formation control on a {grid_size}x{grid_size} grid.\n\n"
            f"Your task:\n"
            f"1. Given a shape description and number of agents, generate {n_agents} coordinates that form the shape\n"
            f"2. Coordinates must be within bounds: 0 to {grid_size-1} for both x and y\n"
            f"3. Spread points evenly to outline the shape (not fill it)\n"
            f"4. Return ONLY a valid JSON array of coordinates, no extra text\n\n"
            f"Format: [[x1, y1], [x2, y2], ..., [xN, yN]]\n\n"
            f"Example for \"circle with 8 agents\":\n"
            f"[[32, 48], [45, 45], [48, 32], [45, 19], [32, 16], [19, 19], [16, 32], [19, 45]]\n\n"
            f"Example for \"star with 10 agents\" (alternating outer/inner points):\n"
            f"[[32, 52], [38, 38], [52, 32], [38, 26], [32, 12], [26, 26], [12, 32], [26, 38], [20, 20], [44, 20]]\n\n"
            f"Example for \"house with 8 agents\" (square base + triangular roof):\n"
            f"[[18, 18], [46, 18], [46, 36], [18, 36], [18, 18], [32, 52], [46, 36], [18, 36]]\n\n"
            f"IMPORTANT:\n"
            f"- All coordinates must be integers between 0 and {grid_size-1}\n"
            f"- Return exactly {n_agents} coordinate pairs\n"
            f"- Output must be valid JSON (use double quotes, proper brackets)\n"
            f"- Do not add any explanations or extra text"
        )
    # default: "standard"
    return (
        f"You are an intelligent coordinates generator for multi-agent formation control on a {grid_size}x{grid_size} grid.\n\n"
        f"Your task:\n"
        f"1. Given a shape description and number of agents, generate {n_agents} coordinates that form the shape\n"
        f"2. Coordinates must be within bounds: 0 to {grid_size-1} for both x and y\n"
        f"3. Spread points evenly to outline the shape (not fill it)\n"
        f"4. Return ONLY a valid JSON array of coordinates, no extra text\n\n"
        f"Format: [[x1, y1], [x2, y2], ..., [xN, yN]]\n\n"
        f"Example for \"circle with 8 agents\":\n"
        f"[[32, 48], [45, 45], [48, 32], [45, 19], [32, 16], [19, 19], [16, 32], [19, 45]]\n\n"
        f"IMPORTANT:\n"
        f"- All coordinates must be integers between 0 and {grid_size-1}\n"
        f"- Return exactly {n_agents} coordinate pairs\n"
        f"- Output must be valid JSON (use double quotes, proper brackets)\n"
        f"- Do not add any explanations or extra text"
    )


def get_completion(prompt, n_agents=4, grid_size=64, model=None, prompt_variant="standard"):
    """
    Sends a prompt to the OpenAI API and returns the response content.

    Args:
        prompt: Natural language description of desired shape
        n_agents: Number of agents/coordinate points to generate
        grid_size: Size of the grid (default 64x64)
        model: OpenAI model name (default: DEFAULT_SHAPE_MODEL)
        prompt_variant: "standard" | "minimal" | "detailed"

    Returns:
        List of coordinates [[x1, y1], [x2, y2], ...]
    """
    model = model or DEFAULT_SHAPE_MODEL
    system_prompt = _build_system_prompt(prompt_variant, n_agents, grid_size)

    try:
        response = client.chat.completions.create(
            model=model,
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


def gen_shape(prompt=None, n_agents=4, grid_size=64, model=None, prompt_variant="standard"):
    """
    Main function to generate shape coordinates.

    Args:
        prompt: Shape description (if None, will prompt user for input)
        n_agents: Number of agents
        grid_size: Grid size
        model: OpenAI model name override (default: DEFAULT_SHAPE_MODEL)
        prompt_variant: "standard" | "minimal" | "detailed"

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
        coordinates = get_completion(prompt, n_agents, grid_size, model=model, prompt_variant=prompt_variant)

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

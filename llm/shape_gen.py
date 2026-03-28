import os
import json
from openai import OpenAI

# The client automatically picks up the OPENAI_API_KEY environment variable
# Alternatively, you can pass the key explicitly: client = OpenAI(api_key="YOUR_API_KEY")
client = OpenAI()

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
            model="gpt-4",  # Use gpt-4, gpt-4-turbo, or gpt-3.5-turbo
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
    import math

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

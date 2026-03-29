#!/usr/bin/env python3
"""
Interactive voice-controlled formation display on Raspberry Pi RGB LED matrix.
Requires: vosk, PIL, numpy, torch, adafruit_blinka_raspberry_pi5_piomatter
"""

import sys
import os
import json
import tempfile
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import vosk
import adafruit_blinka_raspberry_pi5_piomatter as piomatter

from environment.grid_env import parallel_env
from environment.model import Actor, dict_obs_to_tensor
from llm.shape_gen import gen_shape, generate_default_circle


# Configuration
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
ACTOR_MODEL_PATH = "models/actor_mlp_final.pt"  # Using MLP (default architecture)
ACTOR_TYPE = "mlp"  # Set to "cnn" if using CNN model
WIDTH = 64
HEIGHT = 64
STEP_DELAY = 0.1  # Seconds between steps


# Color palette for agents (distinct colors)
AGENT_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (255, 255, 255),  # White
    (128, 255, 0),    # Lime
]

TARGET_COLOR = (64, 64, 64)  # Dim gray for targets
TRAIL_COLOR = (16, 16, 16)   # Very dim trail


def init_matrix():
    """Initialize the RGB LED matrix."""
    geometry = piomatter.Geometry(
        width=WIDTH,
        height=HEIGHT,
        n_addr_lines=5,
        rotation=piomatter.Orientation.Normal
    )

    canvas = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    framebuffer = np.asarray(canvas) + 0

    matrix = piomatter.PioMatter(
        colorspace=piomatter.Colorspace.RGB888Packed,
        pinout=piomatter.Pinout.AdafruitMatrixBonnet,
        framebuffer=framebuffer,
        geometry=geometry
    )

    return matrix, canvas, framebuffer


def listen_once(recognizer):
    """
    Record audio and return recognized text.

    Returns:
        str: Recognized text, or empty string if nothing heard
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmpfile = f.name

    # Record 4 seconds of audio
    subprocess.run([
        "arecord", "-D", "plughw:2,0", "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-t", "wav", "-d", "4", tmpfile
    ], stderr=subprocess.DEVNULL)

    # Process audio
    with open(tmpfile, "rb") as f:
        f.read(44)  # Skip WAV header
        data = f.read()

    text = ""
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()

    os.unlink(tmpfile)
    return text


def get_yes_no(recognizer):
    """
    Listen for yes/no response.

    Returns:
        bool: True for yes, False for no
    """
    while True:
        response = listen_once(recognizer).lower()
        if "yes" in response or "yeah" in response or "correct" in response:
            return True
        elif "no" in response or "nope" in response or "wrong" in response:
            return False
        else:
            print("Please say 'yes' or 'no'")


def extract_number(text):
    """
    Extract a number from text (handles both digits and words).

    Args:
        text: Input text

    Returns:
        int or None: Extracted number or None if not found
    """
    # Map word numbers to digits
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
    }

    # Try to find digits
    words = text.split()
    for word in words:
        if word.isdigit():
            return int(word)
        if word in word_to_num:
            return word_to_num[word]

    return None


def draw_grid(canvas, agent_positions, target_coords, trails=None):
    """
    Draw current state on canvas.

    Args:
        canvas: PIL Image canvas
        agent_positions: Dict mapping agent names to (x, y) positions
        target_coords: List of target [x, y] coordinates
        trails: Optional dict of agent trails (list of positions)
    """
    draw = ImageDraw.Draw(canvas)

    # Clear canvas
    draw.rectangle((0, 0, WIDTH-1, HEIGHT-1), fill=(0, 0, 0))

    # Draw trails if provided
    if trails:
        for agent_idx, trail in enumerate(trails.values()):
            for pos in trail[-10:]:  # Last 10 positions
                if 0 <= pos[0] < WIDTH and 0 <= pos[1] < HEIGHT:
                    canvas.putpixel((pos[0], pos[1]), TRAIL_COLOR)

    # Draw targets (small dim markers)
    for coord in target_coords:
        x, y = coord[0], coord[1]
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            canvas.putpixel((x, y), TARGET_COLOR)

    # Draw agents (bright pixels)
    for agent_idx, (agent_name, pos) in enumerate(agent_positions.items()):
        x, y = pos
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
            # Make agent a 2x2 bright square
            canvas.putpixel((x, y), color)
            if x + 1 < WIDTH:
                canvas.putpixel((x + 1, y), color)
            if y + 1 < HEIGHT:
                canvas.putpixel((x, y + 1), color)
            if x + 1 < WIDTH and y + 1 < HEIGHT:
                canvas.putpixel((x + 1, y + 1), color)


def run_policy_on_matrix(actor, env, target_coords, n_agents, matrix, canvas, framebuffer, max_steps=500):
    """
    Run trained policy and display on LED matrix.

    Args:
        actor: Trained actor network
        env: Environment instance
        target_coords: Target coordinates
        n_agents: Number of agents
        matrix: LED matrix controller
        canvas: PIL canvas
        framebuffer: Numpy framebuffer
        max_steps: Maximum steps to run
    """
    device = 'cpu'  # Use CPU on Raspberry Pi
    obs, info = env.reset(seed=42, target_coords=target_coords)

    # Track trails
    trails = {agent: [] for agent in env.agents}

    print("\nStarting formation display on LED matrix...")
    print("Press Ctrl+C to stop\n")

    step = 0
    success = False

    try:
        while env.agents and step < max_steps:
            # Get current agent positions
            agent_positions = {}
            for agent in env.agents:
                pos = obs[agent]['self_position']
                # Convert from normalized [0, 1] to grid coordinates
                x = int(pos[0] * (WIDTH - 1))
                y = int(pos[1] * (HEIGHT - 1))
                agent_positions[agent] = (x, y)
                trails[agent].append((x, y))

            # Draw current state
            draw_grid(canvas, agent_positions, target_coords, trails)
            framebuffer[:] = np.asarray(canvas)
            matrix.show()

            # Get actions from trained policy
            actions = {}
            for agent in env.agents:
                obs_tensor = dict_obs_to_tensor(obs[agent], device)
                with torch.no_grad():
                    action, _ = actor.get_action(obs_tensor, deterministic=True)
                actions[agent] = action.item()

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step += 1

            # Check if episode ended
            if all(terminations.get(agent, False) for agent in env.possible_agents):
                success = True
                break

            time.sleep(STEP_DELAY)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Final display
    agent_positions = {}
    for agent in env.possible_agents:
        if agent in obs:
            pos = obs[agent]['self_position']
            x = int(pos[0] * (WIDTH - 1))
            y = int(pos[1] * (HEIGHT - 1))
            agent_positions[agent] = (x, y)

    draw_grid(canvas, agent_positions, target_coords, trails)
    framebuffer[:] = np.asarray(canvas)
    matrix.show()

    if success:
        print(f"\n✓ Success! Formation completed in {step} steps")
    else:
        print(f"\n✗ Reached maximum steps ({max_steps})")

    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    env.close()


def main():
    print("="*60)
    print("Interactive Formation Display")
    print("="*60)
    print()

    # Initialize speech recognition
    print("Initializing speech recognition...")
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Error: Vosk model not found at {VOSK_MODEL_PATH}")
        print("Please download: https://alphacephei.com/vosk/models")
        return

    model = vosk.Model(VOSK_MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(model, 16000)
    print("✓ Speech recognition ready\n")

    # Initialize LED matrix
    print("Initializing LED matrix...")
    matrix, canvas, framebuffer = init_matrix()
    print("✓ LED matrix ready\n")

    # Step 1: Get shape
    shape = None
    while shape is None:
        print("What shape would you like? (circle, square, triangle, line, etc.)")
        print("Listening...")

        shape_text = listen_once(recognizer)
        if not shape_text:
            print("Didn't hear anything, please try again\n")
            continue

        print(f"Heard: '{shape_text}'")
        print("Is this correct? (yes/no)")

        if get_yes_no(recognizer):
            shape = shape_text
            print(f"✓ Shape confirmed: {shape}\n")
        else:
            print("Let's try again\n")

    # Step 2: Get number of agents
    n_agents = None
    while n_agents is None:
        print("How many agents? (say a number)")
        print("Listening...")

        number_text = listen_once(recognizer)
        if not number_text:
            print("Didn't hear anything, please try again\n")
            continue

        n_agents = extract_number(number_text)
        if n_agents is None:
            print(f"Couldn't understand number from: '{number_text}'")
            print("Please try again\n")
            continue

        # Limit to reasonable number for display
        if n_agents < 2 or n_agents > 20:
            print(f"Number must be between 2 and 20 (heard: {n_agents})")
            n_agents = None
            continue

        print(f"Heard: {n_agents} agents")
        print("Is this correct? (yes/no)")

        if get_yes_no(recognizer):
            print(f"✓ Number of agents confirmed: {n_agents}\n")
        else:
            n_agents = None
            print("Let's try again\n")

    # Step 3: Generate target coordinates
    print(f"Generating target coordinates for '{shape}' with {n_agents} agents...")
    try:
        target_coords = gen_shape(shape, n_agents=n_agents, grid_size=WIDTH)
        print("✓ Generated using LLM")
    except Exception as e:
        print(f"LLM generation failed: {e}")
        print("Using default circle formation")
        target_coords = generate_default_circle(n_agents, grid_size=WIDTH)

    print(f"Target coordinates: {target_coords}\n")

    # Step 4: Load trained model
    print("Loading trained model...")
    if not os.path.exists(ACTOR_MODEL_PATH):
        print(f"Error: Model not found at {ACTOR_MODEL_PATH}")
        print("Please train a model first using: python main.py --mode train")
        return

    actor = Actor(obs_radius=5).to('cpu')
    actor.load_state_dict(torch.load(ACTOR_MODEL_PATH, map_location='cpu'))
    actor.eval()
    print("✓ Model loaded\n")

    # Step 5: Create environment
    print("Creating environment...")
    env = parallel_env(
        render_mode=None,
        n_agents=n_agents,
        obs_radius=5
    )
    print("✓ Environment ready\n")

    # Step 6: Run policy on LED matrix
    run_policy_on_matrix(actor, env, target_coords, n_agents, matrix, canvas, framebuffer)

    print("\n" + "="*60)
    print("Session complete!")
    print("="*60)


if __name__ == "__main__":
    main()

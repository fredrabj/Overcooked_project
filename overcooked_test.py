from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import random
import time
import cv2
import numpy as np
import pygame.surfarray

# Initalize the Overcooked game
mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, info_level=1, horizon=400)

# Reset environment before we begin
obs = env.reset()
viz = StateVisualizer(tile_size=60, is_rendering_action_probs=False)
grid = mdp.terrain_mtx

while True:
    if env.is_done():
        obs = env.reset()

    joint_action = [
        random.choice(Action.ALL_ACTIONS),
        random.choice(Action.ALL_ACTIONS)
    ]

    obs, reward, done, info = env.step(joint_action)
    frame = viz.render_state(obs, grid=grid)
    array = pygame.surfarray.array3d(frame)  # shape: (width, height, 3), RGB

    # Transpose to (height, width, 3) for OpenCV
    frame_rgb = np.transpose(array, (1, 0, 2))

    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Display with OpenCV
    cv2.imshow("Frame", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
    time.sleep(0.1)

cv2.destroyAllWindows()

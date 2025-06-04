import gym.spaces
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import cv2
import numpy as np
import pygame.surfarray
import gym
from tqdm import tqdm
from overcooked_ai_py.mdp.actions import Action
import random

class RandomAgent:
    def __init__(self):
        return
    
    def get_action(self):
        return random.choice(Action.ALL_ACTIONS)

class GeneralizedOvercooked():

    def __init__(self, layout, info_level=0, horizon=400, fps=10): 
        super().__init__()
        mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(mdp, info_level=info_level, horizon=horizon)
        self.viz = StateVisualizer(tile_size=60, is_rendering_action_probs=False)
        self.grid = mdp.terrain_mtx
        
        # Create Observation space and action space
        dummy_state = mdp.get_standard_start_state()
        dummy_obs = self.env.featurize_state_mdp(dummy_state)[0]
        self.obs_dim = dummy_obs.shape[0]
        self.obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.act_dim = len(Action.ALL_ACTIONS)
        self.act_space = gym.spaces.Discrete(self.act_dim)

        self.obs = None
        self.fps = fps

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.obs = self.env.state
        return self.env.featurize_state_mdp(self.obs)


    def step(self, joint_action):
        self.obs, reward, done, info = self.env.step(joint_action)
        return self.env.featurize_state_mdp(self.obs), reward, done, info


    def render(self):
        if self.obs is None:
            return False

        frame = self.viz.render_state(self.obs, grid=self.grid)
        array = pygame.surfarray.array3d(frame)
        frame_rgb = np.transpose(array, (1, 0, 2))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Overcooked", frame_bgr)
        key = cv2.waitKey(int(1000/self.fps)) & 0xFF
        if key == ord('q'):
            return True 
        return False

    def close(self):
        cv2.destroyAllWindows()


def sample():
    # Initalize the Overcooked game
    env = GeneralizedOvercooked("cramped_room", horizon=1200)
    agent_0 = RandomAgent()
    agent_1 = RandomAgent()

    env.reset()
    done = False
    ep_reward = 0
    while not done:
        if env.render():
            break

        action_0 = agent_0.get_action()
        action_1 = agent_1.get_action()

        joint_action = [action_0, action_1]

        _, reward, done, _ = env.step(joint_action)
        
        ep_reward += reward
    if ep_reward != 0:
        print(ep_reward)

    env.close()

def main():
    sample()


if __name__ == "__main__":
    main()
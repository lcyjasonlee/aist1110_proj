import gym
import pygame
from .the_floor_is_lava import *


class MainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, map_width=9, map_height=15, difficulty=2, render_mode=None, fps=None, trunc=0, seed=None) -> None:

        # 1D vector:
        # player xy, monster xy, freezer&redbull cooldown,
        # then flattened map slice
        self.observation_space = gym.spaces.Box(
            shape=(6+map_height*map_width,),
            low=-1,
            high=trunc
        )

        # 8 walk, 8 destroy, 4 jump, freezer, redbull
        self.action_space = gym.spaces.Discrete(22)

        # actual game
        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.difficulty = difficulty
        self.seed = seed
        self.playground = Playground(self.MAP_WIDTH, self.MAP_HEIGHT, self.difficulty, self.seed)

        # objects for rendering
        self.render_mode = render_mode
        if render_mode == "human":
            self.fps = fps
            self.window = Window(self.playground, self.fps)

        self.trunc = trunc  # truncate after $(trunc) steps
        self.step_count = 0 # no. of steps taken, including invalid ones


    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)    # reset RNG

        self.step_count = 0
        self.playground.reset()

        observation = self.playground.rl_state
        info = {
            "step_count": 0,
            "score": 0
        }

        self._render_frame()

        return observation, info


    def step(self, action) -> tuple:

        self.step_count += 1
        status = self.playground.play(action)[0]

        observation = self.playground.rl_state
        
        if status.success is False and status.score == 0:
            reward = -2
        elif status.score == 0:
            reward = -0.5
        else:
            reward = status.score

        terminated = (status == DEAD_STATUS)
        if self.trunc:
            truncated = (self.step_count >= self.trunc)

        info = {
            "step_count": self.step_count,
            "score": self.playground.score
        }

        self._render_frame()

        return observation, reward, terminated, truncated, info


    def render(self) -> None:
        # outputting frames for training isn't required
        return None


    def _render_frame(self) -> None:
        if self.render_mode == "human":
            self.window.draw()


    def close(self) -> None:
        if self.window is not None:
            pygame.quit()

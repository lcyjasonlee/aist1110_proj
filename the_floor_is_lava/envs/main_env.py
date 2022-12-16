import pygame
import gym
import numpy as np


# handles map creation, path validation
# and updating player+monster positions

# player & monster has very limited capabilities,
# and both are heavily tied to the map,
# might as well put them into 1 class
class Map:
    
    player_actions = None
    monster_actions = None
    
    
    def __init__(self) -> None:
        # the map maybe better stored by lists rather than ndarray,
        # as the map grows as the player moves forward
        
        # we may hardcode the first few rows
        # and dynamically generate the rest
        self.map = None
        
        # cooldown of player's freezer
        self.freezer_cooldown = 0


    # return 2d array representing the state:
    # 1st row is for player+monster stats,
    # the rest is the actual observable map
    
    # this time we may use ndarray,
    # as observations are read-only
    
    # used for rendering and providing states for RL algo
    def state(self) -> np.ndarray:
        pass
    
    
    # return lists of actions a player/monster can take,
    # for internal use only
    
    # a player may walk, jump, use freezer, destroy platforms
    # freezer has cooldown
    def _valid_action_player(self) -> list:
        pass
    
    # a monster may walk/jump
    def _valid_action_monster(self) -> list:
        pass
    
    
    # take a random action for the monster
    # among all valid actions,
    # prefer actions that can move the monster forward
    
    # for internal use:
    # called right after player makes a move
    def _walk_monster(self) -> None:
        valid_actions = self.valid_action_monster()
        

    # execute player actions and update the map,
    # player is always placed on center row
    
    # freezer cooldown -= 1 after every action,
    # unless action=use freezer when cooldown > 0
    
    # return no. of platforms advanced,
    # can be -ve when walking backwards
    def play(self, action) -> int:
        self._walk_monster()
    

# handles pygame window rendering
# including sprites, map, score display, etc.
class Window:
    font = pygame.font.get_fonts("")
    
    def __init__(self) -> None:
        pygame.init()
        
        self.win = pygame.display.set_mode((720, 1280)) # resolution pending
        self.clock = pygame.time.Clock()

        # TODO: more init arguments & variables


    # most of the rendering belongs to here
    def draw(self) -> None:
        self.clock.tick(framerate=30)   # fps pending
        
        self.win.fill((0,0,0))
        
        # TODO: rendering code
        
        pygame.display.update()


class MainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}


    def __init__(self, render_mode=None) -> None:
        
        # 1st row for storing player/monster info
        # the rest for the actual map
        self.map = Map()
        
        self.obs_space = gym.spaces.Box(shape=(16, 9))
        self.act_space = None

        # objects for rendering
        self.window = Window()
        
        
    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)    # reset RNG

        # TODO: resetting code
        
        observation = self.map
        info = dict()   # no extra info
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    
    def step(self, action) -> tuple:
        
        # TODO: step code + assign variables below
        observation = self.map.state()
        reward = None
        terminated = None
        truncated = None
        
        info = dict()
        
        
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info


    def render(self) -> None:
        # outputting frames for training isn't required
        return None


    def _render_frame(self) -> None:
        if self.render_mode == "human":
            
            if self.window is None:
                self.window = Window()
                
            self.window.draw()


    def close(self) -> None:
        if self.window is not None:
            pygame.quit()
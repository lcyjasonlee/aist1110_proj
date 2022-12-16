import pygame
import gym
import numpy as np

key_to_action = {
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3,
    'space': 4, # jump
    'k': 5, # freezer
    ';': 6 # redbull
}

key_to_action2 = {
    (pygame.K_UP, pygame.K_a, pygame.K_KP8): 0,
    (pygame.K_DOWN, pygame.K_s, pygame.K_KP6): 1,
    (pygame.K_LEFT, pygame.K_a, pygame.K_KP4) : 2,
    (pygame.K_RIGHT, pygame.K_d, pygame.K_KP6): 3,
    (pygame.K_SPACE, ): 4, # jump
    (pygame.K_k, ): 5, # freezer
    (pygame.K_SEMICOLON): 6 # redbull
}

action_to_direction = {
    0: [0, -1],
    1: [0, 1],
    2: [-1, 0],
    3: [1, 0]
}

# handles map creation, path validation
# and updating player+monster positions

# player & monster has very limited capabilities,
# and both are heavily tied to the map,
# might as well put them into 1 class
class Map:

    #player_actions = None
    #monster_actions = None

    def __init__(self) -> None:
        # the map maybe better stored by lists rather than ndarray,
        # as the map grows as the player moves forward

        # we may hardcode the first few rows
        # and dynamically generate the rest
        self.map = [[0 for i in range(9)] for j in range(15)]

        # initial platforms at center
        self.init_platform_size = 5
        for j in range(len(self.map)//2 - self.init_platform_size//2, len(self.map)//2 + self.init_platform_size//2 + 1):
            for i in range(len(self.map[0])//2 - self.init_platform_size//2, len(self.map[0])//2 + self.init_platform_size//2  + 1):
                self.map[j][i] = 1

        # cooldown of player's freezer
        #self.freezer_cooldown = 0


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
    #def _valid_action_player(self) -> list:
     #   pass

    # a monster may walk/jump
    #def _valid_action_monster(self) -> list:
     #   pass


    # take a random action for the monster
    # among all valid actions,
    # prefer actions that can move the monster forward

    # for internal use:
    # called right after player makes a move
    #def _walk_monster(self) -> None:
     #   valid_actions = self.valid_action_monster()


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
    font = pygame.font.get_fonts()

    def __init__(self) -> None:
        self.win_size = (271, 541) # resolution pending
        self.win = pygame.display.set_mode(self.win_size)
        self.grid_size = 30
        self.canvas = pygame.Surface(self.win_size)
        self.all_sprites = pygame.sprite.Group()

        self.clock = pygame.time.Clock()

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("The Floor is Lava")


    # most of the rendering belongs to here
    def draw(self, grids) -> None:
        self.clock.tick(15)   # fps pending

        # TODO: rendering code

        self.canvas.fill("darkorange2") # background lava

        for i in range(len(grids)+1): # draw horizontal line
            pygame.draw.line(self.canvas, "azure3", (0, (i+3)*self.grid_size), (len(grids[0])*self.grid_size, (i+3)*self.grid_size))

        for i in range(len(grids[0])+1): # draw vertical line
            pygame.draw.line(self.canvas, "azure3", (i*self.grid_size, 3*self.grid_size), (i*self.grid_size, (len(grids)+3)*self.grid_size))

        # draw platforms
        for i in range(len(grids)):
            for j in range(len(grids[0])):
                if grids[i][j] == 1:
                    pygame.draw.rect(self.canvas, "antiquewhite1", pygame.Rect((j*self.grid_size + 1, (i+3)*self.grid_size + 1), (self.grid_size - 1, self.grid_size - 1)))

        self.all_sprites.draw(self.canvas)

        self.win.blit(self.canvas, self.canvas.get_rect())
        pygame.display.flip()


class Action():
    def __init__(self, max_jump=2, max_freeze=3, freezer_cooldown=30):
        self.max_jump = max_jump # diamond shape from player
        self.max_freeze = max_freeze

        # relative positions of platforms freezable (3*3 diamond shape)
        self.can_freeze = \
                        [[i, j] for i in range(-self.max_freeze+1, self.max_freeze) for j in range(-self.max_freeze+1, self.max_freeze)
                        if (i, j) != (0, 0) and (abs(i), abs(j)) != (self.max_freeze-1, self.max_freeze-1)] + \
                        [[0, self.max_freeze], [0, -self.max_freeze], [self.max_freeze, 0], [-self.max_freeze, 0]]

        self.freezer_cooldown = freezer_cooldown
        self.until_freezer = freezer_cooldown

    # move the location in a direction
    @staticmethod
    def _move(loc: list[int], dir: list[int], max_x: int = 9, max_y: int = 15) -> list[int]:
        loc = [i+j for i, j in zip(loc, dir)]

        if loc[0] < 0:
            loc[0] = 0
        if loc[0] >= max_x:
            loc[0] = max_x-1
        if loc[1] < 0:
            loc[1] = 0
        if loc[1] >= max_y:
            loc[1] = max_y - 1

        return loc

    # move to a direction
    def walk(self, loc: list[int], dir: list[int]) -> list[int]:
        return self._move(loc, dir)

    # move 2 blocks to a direction
    def jump(self, loc: list[int], dir: list[int]) -> list[int]:
        return self._move(loc, map(lambda x: x*self.max_jump, dir))

    @property
    def _has_freeze(self) -> bool:
        return self.until_freezer == 0

    # freeze platforms
    def freezer(self, pos: list[int], grids: list[list[int]]) -> None:
        if not self._has_freeze:
            return
        print("Halo")

        for dir in self.can_freeze:
            x, y = [i+j for i, j in zip(pos, dir)]
            if y in range(0, 15) and x in range(0, 9) and grids[y][x] == 0:
                grids[y][x] = 1

        self.until_freezer = self.freezer_cooldown # reset cooldown


    @staticmethod
    def is_alive(pos: list[int], grids: list[list[int]]) -> bool:
        return grids[pos[1]][pos[0]] == 1 # on platform


class Player(pygame.sprite.Sprite, Action):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        Action.__init__(self)

        # rows = 15; cols = 9; grid_size = 30
        self.image = pygame.Surface([30//2, 30//2]) # change to an image later
        self.image.fill("red")
        self.rect = self.image.get_rect(center=((9//2 + 0.5) * 30 + 0.5, (15//2 + 0.5 + 3) * 30 + 0.5))

        self.location = [9//2, 15//2] # centre of map, (x, y)


    def walk(self, dir: list[int]) -> None:
        self.location = super().walk(self.location, dir)
        self.rect.move_ip(*map(lambda x: x*30, dir))

    def jump(self, dir: list[int]) -> None:
        self.location = super().walk(self.location, list(map(lambda x: x*2, dir)))
        self.rect.move_ip(*map(lambda x: x*30*2, dir))

    def freezer(self, pos: list[int], grids: list[list[int]]) -> None | list[list[int]]:
        return super().freezer(pos, grids)


class Monster(pygame.sprite.Sprite, Action):
    pass



class MainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}


    def __init__(self, render_mode=None) -> None:

        self.map = Map()
        self.player = Player()
        self.monster = None

        # 1st row for storing player/monster info
        # the rest for the actual map
        self.obs_space = gym.spaces.Box(shape=(16, 9))
        # actions (walk, jump, freezer, redbull), x (-1/-2, +1/+2), y (-1/-2, +1,+2)
        self.act_space = gym.spaces.MultiDiscrete(shape=[5, 2, 2])

        # objects for rendering
        self.window = Window()


    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)    # reset RNG

        # TODO: resetting code

        self.map = Map()
        self.player = Player()
        self.monster = None

        observation = self.map.map
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


if __name__ == "__main__":
    grids = Map()
    win = Window()
    win.__init__()

    player = Player()
    win.all_sprites.add(player)

    running = True
    jump = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key not in key_to_action:
                    continue

                key = key_to_action[key]

                if key == 4:
                    jump = True

                if key == 5:
                    player.freezer(player.location, grids.map)

                elif key in {0, 1, 2, 3}:
                    if jump:
                        player.jump(action_to_direction[key])
                        jump = False
                    else:
                        player.walk(action_to_direction[key])

        if not player.is_alive(player.location, grids.map):
            player.kill()
            running = False

        win.draw(grids.map)

    pygame.quit()

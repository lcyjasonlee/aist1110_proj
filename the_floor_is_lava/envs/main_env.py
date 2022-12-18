import pygame
import gym
import numpy as np
from itertools import product
from collections import namedtuple


MAP_HEIGHT = 15
MAP_WIDTH = 9
GRID_SIZE = 32

key_to_action = {
    'w': 0, # up
    's': 1, # down
    'a': 2, # left
    'd': 3, # right
    'q': 4, # up-left
    'e': 5, # up-right
    'z': 6, # down-left
    'c': 7, # down-right
    'space': 8, # jump
    'l': 9, # freezer
    ';': 10, # redbull
    'k': 11, # destroy
}

key_to_action2 = {
    (pygame.K_UP, pygame.K_a, pygame.K_KP8): 0, # up
    (pygame.K_DOWN, pygame.K_s, pygame.K_KP6): 1, # down
    (pygame.K_LEFT, pygame.K_a, pygame.K_KP4) : 2, # left
    (pygame.K_RIGHT, pygame.K_d, pygame.K_KP6): 3, # right
    (pygame.K_q, ): 4, # up-left
    (pygame.K_e, ): 5, # up-right
    (pygame.K_z, ): 6, # down-left
    (pygame.K_c, ): 7, # down-right
    (pygame.K_SPACE, pygame.K_j): 8, # jump
    (pygame.K_l, ): 9, # freezer
    (pygame.K_SEMICOLON, ): 10, # redbull
    (pygame.K_k, ): 11, # destroy
}

# make coordinates objects for better readability,
# handles coordinate operations too
class Coordinate:
    x_max, y_max = MAP_WIDTH-1, MAP_HEIGHT-1


    # must use kwarg, for readability in object creation
    def __init__(self, *, x: int=None, y: int=None) -> None:
        if x is None or y is None:
            raise ValueError("x, y Coordinate Not Specified")
        else:
            self.x = x
            self.y = y


    # c: Coordinate object
    def __add__(self, c):
        newx, newy = self.x + c.x, self.y + c.y
        return Coordinate(x=newx, y=newy)


    def __sub__(self, c):
        return self + (-1 * c)


    def __mul__(self, op: int|float):
        EPSILON = 0.01
        newx, newy = self.x * op, self.y * op
        newx_int, newy_int = round(newx), round(newy)

        if abs(newx_int - newx) > EPSILON or abs(newy_int - newy) > EPSILON:
            raise ValueError("Non-Integer Coordinate")
        else:
            return Coordinate(x=newx_int, y=newy_int)


    def __truediv__(self, op: int|float):
        return self * (1 / op)


    def __floordiv__(self, op: int):
        return Coordinate(
            x=int(self.x // op),
            y=int(self.y // op)
            )


    # index=True: tuple is in order of matrix indices
    # i.e. (a, b) meant to be interpreted as matrix[a][b]
    def coord(self, *, index=None):
        if index is True:
            return (self.y, self.x)
        elif index is False:
            return (self.x, self.y)
        else:
            raise ValueError("Coordinate Order Not Specified")


action_to_direction = (
    Coordinate(x=0, y=-1),
    Coordinate(x=0, y=1),
    Coordinate(x=-1, y=0),
    Coordinate(x=1, y=0),
    Coordinate(x=-1, y=-1),
    Coordinate(x=1, y=-1),
    Coordinate(x=-1, y=1),
    Coordinate(x=1, y=1),
)


class Map:
    """
    handles map creation and path validation
    """
    def __init__(self) -> None:
        # the map maybe better stored by lists rather than ndarray,
        # as the map grows as the player moves forward

        # we may hardcode the first few rows
        # and dynamically generate the rest
        self.map = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)] # actual map

        # initial platforms at center
        # TODO: move this to playground
        self.init_platform_size = 5
        for j in range(MAP_HEIGHT//2 - self.init_platform_size//2, MAP_HEIGHT//2 + self.init_platform_size//2 + 1):
            for i in range(MAP_WIDTH//2 - self.init_platform_size//2, MAP_WIDTH//2 + self.init_platform_size//2  + 1):
                self.map[j][i] = 1

        self.grids = self.map[:] # observable area
        self.grid_centre = MAP_HEIGHT // 2 # centre row


    def shift(self, shift: int) -> None:
        # upward = +(1), downward = -(1)
        if shift == 0:
            return

        if shift > 0:
            n = -(self.grid_centre - MAP_HEIGHT//2 - shift) # number of new rows to be generated
            if n > 0:
                new_row = self.gen_platform(n)
                self.map = new_row + self.map
            else:
                self.grid_centre -= shift # if not, shift current row up by (shift)
            self.grids = self.map[self.grid_centre - MAP_HEIGHT//2:self.grid_centre + MAP_HEIGHT//2 + 1]
            print("Shifted upward")
            print(n, self.grid_centre)
            print(self.grids)

        else:
            bottom = self.grid_centre + MAP_HEIGHT//2
            n = bottom - shift - len(self.map) - 1
            if n > 0:
                new_row = self.gen_platform(n)
                self.map = self.map + new_row
            else:
                self.grid_centre -= shift
            self.grids = self.map[self.grid_centre - MAP_HEIGHT//2:self.grid_centre + MAP_HEIGHT//2 + 1]
            print("Shifted Downward")
            print(n, self.grid_centre)


    def update_map(self):
        self.map[self.grid_centre - MAP_HEIGHT//2: self.grid_centre + MAP_HEIGHT//2 + 1] = self.grids


    @staticmethod
    def gen_platform(n: int):
        return [[1 for i in range(MAP_WIDTH)] for j in range(n)]
    
    @staticmethod
    def out_of_bound(c: Coordinate):
        if c.x < 0 or c.x >= MAP_WIDTH or c.y < 0:
            return True
        else:
            return False


# score = score given to player
# success + score determines reward
# e.g. even if score = 0, can still have -ve reward
Status = namedtuple("Status", ["success", "score"])


class Entity:

    def __init__(self, start_loc: Coordinate):
        self.location = start_loc
    
    # move 1 block up/down/left/right
    # loc is passed by reference
    def walk(self, dir: Coordinate, map: Map) -> Status: # coord, dy
        newloc = self.location + dir
        
        if Map.out_of_bound(newloc):
            return Status(False, 0)
        else:
            self.location = newloc
            return Status(True, dir.y)

    # can jump 2 platforms to up/down/left/right, but not diagonal
    def jump(self, dir: Coordinate, map: Map) -> Status: # coord, dy
        newloc = self.location + dir * 2
        
        if Map.out_of_bound(newloc):
            return Status(False, 0)
        else:
            self.location = newloc
            return Status(True, dir.y * 2)

    def destroy(self, dir: Coordinate, map: Map) -> None:
        target = self.location + dir
        
        if Map.out_of_bound(target):
            return Status(False, 0)
        
        if map.map[target.y][target.x] == 0:
            return Status(False, 0)
        else:
            map.map[target.y][target.x] = 0
            return Status(True, 0)

    def is_alive(self, map: Map) -> bool:
        return map.map[self.location.y][self.location.x] == 1 # on platform


class Player(Entity):
    def __init__(self, coordinate, max_freeze=3, freezer_cooldown=7, redbull_cooldown=0):
        
        Entity.__init__(self, coordinate)
        
        self.max_freeze = max_freeze

        _iter1= range(-max_freeze, max_freeze+1)
        _iter2 = range(-max_freeze, max_freeze+1)

        # manhattan distance
        self.can_freeze = [
            Coordinate(x=i, y=j)
            for i, j in product(_iter1, _iter2)
            if abs(i)+abs(j) <= max_freeze
            and not (i == 0 and j == 0)
            ]

        self.freezer_cooldown = freezer_cooldown    # constant
        self.until_freezer = freezer_cooldown

        self.redbull_cooldown = redbull_cooldown    # constant
        self.until_redbull = redbull_cooldown

    @property
    def has_freezer(self) -> bool:
        return self.until_freezer == 0
    
    @property
    def has_redbull(self) -> bool:
        return self.redbull_cooldown == 0
    
    # freeze platforms
    def freezer(self, map: Map) -> Status: 
        if not self.has_freezer:
            return Status(False, 0)

        count = 0
        for freeze_coord in self.can_freeze:
            try:
                coord = self.location + freeze_coord
                if map.map[coord.y][coord.x] == 0:
                    map.map[coord.y][coord.x] = 1
                    count += 1
            except IndexError:
                pass    # only freeze what's possible

        self.until_freezer = self.freezer_cooldown # reset cooldown
        
        # using freezer when it has no effect = failure
        return Status(bool(count), 0)


    def redbull(self, map: Map) -> Status: 
        if not self.has_redbull:
           return Status(False, 0)

        # TODO: better way to select new platform
        x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)
        while map.map[y][x] == 0 or (x, y) == (self.location.x, self.location.y): # lava | current pos
            x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)

        self.location = Coordinate(x=x, y=y)
        return Status(True, self.location.y - y)

    
    
    # a player may walk, jump, use freezer, destroy platforms
    # freezer has cooldown
    def valid_action(self) -> list:
       pass


class Monster(Entity):
    def __init__(self):
        Entity.__init__(self, Coordinate(x=MAP_WIDTH//2 - 1, y=MAP_HEIGHT//2 - 1))

    # a monster may walk/jump
    def valid_action(self) -> list:
       pass
    
    # take a random action for the monster
    # among all valid actions,
    # prefer actions that can move the monster forward
    def step(self) -> None:
        pass
    

class Playground:
    
    # 0-7: walk (see action_to_direction)
    # 8-15: destroy
    # 16-19: jump (up down left right)
    # 20: freezer, 21: redbull
    
    def __init__(self) -> None:
        self.map = Map()
        self.player = Player(Coordinate(x=MAP_WIDTH//2, y=MAP_HEIGHT//2))
        self.monster = Monster()
        
    def play(self, action: int) -> int:
        # match case can't match range yetw
        if action in range(0, 7+1):
            self.player.walk(action_to_direction[action], self.map)
        elif action in range(8, 15+1):
            self.player.destroy(action_to_direction[action-8], self.map)
        elif action in range(16, 19+1):
            self.player.jump(action_to_direction[action-16], self.map)
        elif action == 20:
            self.player.freezer(self.map)
        elif action == 21:
            self.player.redbull(self.map)
        else:
            raise ValueError("Unknown Action")
        
        
    # return 2d array representing the state:
    # 1st row is for player+monster stats,
    # the rest is the actual observable map

    # this time we may use ndarray,
    # as observations are read-only

    # used for rendering and providing states for RL algo
    def state(self) -> np.ndarray:
        pass
        

class Window:
    """
    handles pygame window rendering,
    including sprites, map, score display, etc.
    """
    
    font = pygame.font.get_fonts()

    def __init__(self, playground: Playground) -> None:
        
        self.playground = playground
        
        self.win_size = (MAP_WIDTH * GRID_SIZE + 1, (MAP_HEIGHT+2) * GRID_SIZE + 1) # resolution pending
        self.win = pygame.display.set_mode(self.win_size)

        # self.grid_size = GRID_SIZE
        self.lava_image = pygame.image.load("assets/lava.png").convert()
        self.platform_image = pygame.image.load("assets/platform.png").convert()
        self.platform_lip_image = pygame.image.load("assets/platform_lip.png").convert_alpha()
        self.player_image = pygame.image.load("assets/player.png").convert_alpha()
        self.monster_image = pygame.image.load("assets/monster.png").convert_alpha()
        
        self.grid_size = self.lava_image.get_height()

        self.surface = pygame.Surface(self.win_size)

        pygame.init()
        self.clock = pygame.time.Clock()

        pygame.display.init()
        pygame.display.set_caption("The Floor is Lava")


    # most of the rendering belongs to here
    def draw(self, grids) -> None:
        self.clock.tick(15)   # fps pending

        # TODO: rendering code

        self.surface.fill("darkorange2") # background colour

        # draw grids of lava / platform
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                topleft = j*self.grid_size, i*self.grid_size

                if grids[i][j] == 0: # lava
                    self.surface.blit(self.lava_image, self.lava_image.get_rect(topleft=topleft))
                    if i-1 >= 0 and grids[i-1][j] == 1: # platform above
                        self.surface.blit(self.platform_lip_image, self.lava_image.get_rect(topleft=topleft))
                else: # platform
                    self.surface.blit(self.platform_image, self.lava_image.get_rect(topleft=topleft))

        # self.all_sprites.draw(self.surface)
        
        player_image_x = self.playground.player.location.x * self.grid_size - 1
        player_image_y = self.playground.player.location.y * self.grid_size - self.grid_size//3
        self.surface.blit(self.player_image, (player_image_x, player_image_y))

        self.win.blit(self.surface, self.surface.get_rect())
        pygame.display.flip()


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

        observation = self.map.state()  # state is more than the map itself
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

    playground = Playground()
    win = Window(playground=playground)

    # TODO: change reference later
    map = playground.map
    player = playground.player
    # win.all_sprites.add(player)

    is_jump = False
    is_destroy = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

            if event.type == pygame.KEYDOWN:
                try:
                    key = key_to_action[pygame.key.name(event.key)]
                    if key == 8:
                        is_jump = True
                        is_destroy = False

                    elif key == 9:
                        player.freezer(map)
                        is_jump = False
                        is_destroy = False

                    elif key == 10:
                        player.redbull(map)
                        is_jump = False
                        is_destroy = False

                    elif key == 11:
                        is_jump = False
                        is_destroy = True

                    else:
                        dir = action_to_direction[key]
                        if is_jump:
                            player.jump(dir, map)
                            is_jump = False
                        elif is_destroy:
                            player.destroy(dir, map)
                            is_destroy = False
                        elif not is_jump and not is_destroy:
                            player.walk(dir, map)

                    #print(player.reward)

                except KeyError:
                    continue

        if not player.is_alive(map):
            # player.kill()
            # player.reward -= 10
            running = False

        win.draw(map.grids)
        #map.update_map()

    pygame.quit()

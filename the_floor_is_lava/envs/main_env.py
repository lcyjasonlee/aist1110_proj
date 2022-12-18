import pygame
import gym
import numpy as np
from itertools import product


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

        if newx > self.x_max or newx < 0 or newy > self.y_max or newy < 0:
            raise ValueError("Coordinate Out of Bound")
        else:
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
            return (self.x, self.x)
        else:
            raise ValueError("Coordinate Order Not Specified")


action_to_direction = {
    0: Coordinate(x=0, y=-1),
    1: Coordinate(x=0, y=1),
    2: Coordinate(x=-1, y=0),
    3: Coordinate(x=1, y=0),
    4: Coordinate(x=-1, y=-1),
    5: Coordinate(x=1, y=-1),
    6: Coordinate(x=-1, y=1),
    7: Coordinate(x=1, y=1),
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
        self.map = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)] # actual map

        # initial platforms at center
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
        self.win_size = (MAP_WIDTH * GRID_SIZE + 1, (MAP_HEIGHT+2) * GRID_SIZE + 1) # resolution pending
        self.win = pygame.display.set_mode(self.win_size)

        # self.grid_size = GRID_SIZE
        self.lava_image = pygame.image.load("assets/lava.png").convert()
        self.platform_image = pygame.image.load("assets/platform.png").convert()
        self.platform_lip_image = pygame.image.load("assets/platform_lip.png").convert_alpha()
        self.platform_lip_image.set_colorkey((255, 255, 255))
        
        self.grid_size = self.lava_image.get_height()

        self.surface = pygame.Surface(self.win_size)
        self.all_sprites = pygame.sprite.Group()

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

        self.all_sprites.draw(self.surface)

        self.win.blit(self.surface, self.surface.get_rect())
        pygame.display.flip()


class Action():
    def __init__(self, max_freeze=3, freezer_cooldown=7, redbull_cooldown=0):

        # jump is meant to be slightly better than walk with drawbacks,
        # so jump range need not be variable
        self.max_freeze = max_freeze

        _iter1= range(-max_freeze, max_freeze+1)
        _iter2 = range(-max_freeze, max_freeze+1)

        # manhattan distanceaaa
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


    # move 1 block up/down/left/right
    # loc is passed by reference
    @staticmethod
    def walk(loc: Coordinate, dir: Coordinate) -> tuple[Coordinate, int]: # coord, dy
        return loc + dir, -dir.y

    # can jump 2 platforms to up/down/left/right, but not diagonal
    @staticmethod
    def jump(loc: Coordinate, dir: Coordinate) -> tuple[Coordinate, int]: # coord, dy
        return loc + (dir*2), -dir.y*2


    @property
    def _has_freezer(self) -> bool:
        return self.until_freezer == 0

    # freeze platforms
    def freezer(self, pos: Coordinate, grids: list[list[int]]) -> bool: # none | reward
        #if not self._has_freezer:
        #    return False # if cooldown not over, return negative reward (-1)

        for freeze_coord in self.can_freeze:
            try:
                coord = pos + freeze_coord
                grids[coord.y][coord.x] = 1
            except ValueError:
                pass    # only freeze what's possible

        self.until_freezer = self.freezer_cooldown # reset cooldown

        return True


    @property
    def _has_redbull(self) -> bool:
        return self.redbull_cooldown == 0

    def redbull(self, pos: Coordinate, grids: list[list[int]]) -> tuple[Coordinate, int]: # coord, dy
        #if not self._has_redbull:
        #    return

        x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)
        while grids[y][x] == 0 or (x, y) == (pos.x, pos.y): # lava | current pos
            x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)

        return Coordinate(x=x, y=y), pos.y-y


    @staticmethod
    def destroy_grid(loc: Coordinate, dir: Coordinate, grids: list[list[int]]) -> None:
        target = loc + dir
        grids[target.y][target.x] = 0


    @staticmethod
    def is_alive(pos: Coordinate, grids: list[list[int]]) -> bool:
        return grids[pos.y][pos.x] == 1 # on platform


class Player(pygame.sprite.Sprite, Action):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        Action.__init__(self)

        self.image = pygame.image.load("assets/player.png").convert_alpha()
        # self.image = pygame.Surface([GRID_SIZE//2, GRID_SIZE//2]) # change to an image later
        # self.image.fill("red")
        self.rect = self.image.get_rect(center=((MAP_WIDTH//2 + 0.5) * GRID_SIZE -1 , (MAP_HEIGHT//2 + 0.5) * GRID_SIZE -7))

        self.location = Coordinate(x=MAP_WIDTH//2, y=MAP_HEIGHT//2) # centre of map, (x, y)
        self.reward = 0


    def walk(self, dir: Coordinate, map: Map) -> None:
        self.location, dy = super().walk(self.location, dir)
        self.reward += dy

        self.rect.move_ip(dir.x * GRID_SIZE, dir.y * GRID_SIZE)
        #map.shift(dy)

        #print("Walk to", self.location.coord)


    def jump(self, dir: Coordinate, map: Map) -> None:
        if abs(dir.x) == 1 and abs(dir.y) == 1: # diagonal jump
            return

        self.location, dy = super().jump(self.location, dir)
        self.reward += dy

        self.rect.move_ip(dir.x * GRID_SIZE * 2, dir.y * GRID_SIZE * 2)
        # map.shift(dy)

        #print("Jump to", self.location.coord)


    def freezer(self, grids: list[list[int]]) -> None | list[list[int]]:
        if not super().freezer(self.location, grids):
            self.reward -= 1

    def redbull(self, grids: list[list[int]]) -> None:
        self.location, dy = super().redbull(self.location, grids)
        self.reward += dy

        self.rect.center = ((self.location.x + 0.5) * GRID_SIZE + 0.5, (self.location.y + 0.5 + 3) * GRID_SIZE + 0.5)

        #print("Redbull to", self.location.coord)

    def destroy_grid(self, dir: Coordinate, grids: list[list[int]]) -> None:
        return super().destroy_grid(self.location, dir, grids)

    def is_alive(self, grids: list[list[int]]) -> bool:
        return super().is_alive(self.location, grids)


class Monster(pygame.sprite.Sprite, Action):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        Action.__init__(self)

        self.image = None # change to an image later
        self.rect = self.image.get_rect(center=None)

        self.location = None # centre of map, (x, y)


    def walk(self, dir: Coordinate) -> None:
        self.location = super().walk(self.location, dir)
        self.rect.move_ip(dir.x * GRID_SIZE, dir.y * GRID_SIZE)


    def jump(self, dir: Coordinate) -> None:
        self.location = super().jump(self.location, dir)
        self.rect.move_ip(dir.x * GRID_SIZE * 2, dir.y * GRID_SIZE * 2)


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
    map = Map()
    win = Window()
    # win.__init__()

    player = Player()
    win.all_sprites.add(player)

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
                        player.freezer(map.grids)
                        is_jump = False
                        is_destroy = False

                    elif key == 10:
                        player.redbull(map.grids)
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
                            player.destroy_grid(dir, map.grids)
                            is_destroy = False
                        elif not is_jump and not is_destroy:
                            player.walk(dir, map)

                    #print(player.reward)

                except KeyError:
                    continue

        if not player.is_alive(map.grids):
            player.kill()
            player.reward -= 10
            running = False

        win.draw(map.grids)
        #map.update_map()

    pygame.quit()

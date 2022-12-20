import pygame
import gym
import numpy as np
from itertools import product
from collections import namedtuple

MAP_HEIGHT = 15
MAP_WIDTH = 9

# 0-7: walk (w/ action_to_direction)
# 8-15: destroy (w/ action_to_direction)
# 16-19: jump (w/ action_to_direction)
# 20: freezer, 21: redbull
keys_to_action = {
    ("w", "8"): 0, # walk; up
    ("s", "2"): 1, # walk; down
    ("a", "4"): 2, # walk; left
    ("d", "6"): 3, # walk; right
    ("q", "7"): 4, # walk; up-left
    ("e", "9"): 5, # walk; up-right
    ("z", "1"): 6, # walk; down-left
    ("c", "3"): 7, # walk; down-right

    ("kw", "k8"): 8, # destroy; up
    ("ks", "k2"): 9, # destroy; down
    ("ka", "k4"): 10, # destroy; left
    ("kd", "k6"): 11, # destroy; right
    ("kq", "k7"): 12, # destroy; up-left
    ("ke", "k9"): 13, # destroy; up-right
    ("kz", "k1"): 14, # destroy; down-left
    ("kc", "k3"): 15, # destroy; down-right

    (" w", " 8", " jw", "j8"): 16, # jump; up
    (" s", " 2", "js", "j2"): 17, # jump; down
    (" a", " 4", "ja", "j4"): 18, # jump; left
    (" d", " 6", "jd", "j6"): 19, # jump; right

    "l": 20, # freezer
    ";": 21, # redbull

}


# make coordinates objects for better readability,
# handles coordinate operations too

# right = +x, up = +y
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
        return self + c*(-1)


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


    def __eq__(self, c) -> bool:
        return (self.x == c.x) and (self.y == c.y)


    # index=True: tuple is in order of matrix indices
    # i.e. (a, b) meant to be interpreted as matrix[a][b]
    def coord(self, *, index=None):
        if index is True:
            return (self.y, self.x)
        elif index is False:
            return (self.x, self.y)
        else:
            raise ValueError("Coordinate Order Not Specified")

OFF_SCREEN = Coordinate(x=-1, y=-1)

class Map:
    """
    handles map creation and path validation
    """
    def __init__(self) -> None:
        # the map maybe better stored by lists rather than ndarray,
        # as the map grows as the player moves forward

        # we may hardcode the first few rows
        # and dynamically generate the rest
        self.grids = [[0 for i in range(MAP_WIDTH)] for j in range(MAP_HEIGHT)] # actual map

        # TODO: move this to playground
        # generate initial platforms
        self.init_platform_size = 5
        for j in range(MAP_HEIGHT//2 - self.init_platform_size//2, MAP_HEIGHT):
            for i in range(MAP_WIDTH//2 - self.init_platform_size//2, MAP_WIDTH//2 + self.init_platform_size//2 + 1):
                self.grids[j][i] = 1


    # append n new platforms to grids
    def expand(self, n: int) -> None:
        for i in range(n):
            self.grids += self.gen_platform()

    @staticmethod
    def gen_platform() -> list[list[int]]:
        #TODO: better way to generate new platforms
        # may generate a chunk of new rows rather than row-by-row
        return [[0, 0, 1, 1, 1, 1, 1, 0, 0]]

    @staticmethod
    def out_of_bound(c: Coordinate):
        return c.x < 0 or c.x >= MAP_WIDTH or c.y < 0


# score = score given to player
# success + score determines reward
# e.g. even if score = 0, can still have -ve reward
Status = namedtuple("Status", ["success", "score"])


class Entity:

    def __init__(self, start_loc: Coordinate):
        self.location = start_loc

    # move 1 block up/down/left/right
    # loc is passed by reference
    def walk(self, dir: Coordinate) -> Status:
        newloc = self.location + dir

        if Map.out_of_bound(newloc):
            return Status(False, 0)
        else:
            self.location = newloc
            return Status(True, dir.y)

    # can jump 2 platforms to up/down/left/right, but not diagonal
    def jump(self, dir: Coordinate) -> Status:
        newloc = self.location + dir * 2

        if Map.out_of_bound(newloc):
            return Status(False, 0)
        else:
            self.location = newloc
            return Status(True, dir.y * 2)

    def destroy(self, dir: Coordinate, m: Map) -> None:
        target = self.location + dir

        if Map.out_of_bound(target):
            return Status(False, 0)

        if m.grids[target.y][target.x] == 0:
            return Status(False, 0)
        else:
            m.grids[target.y][target.x] = 0
            return Status(True, 0)


    def is_alive(self, m: map) -> bool:
        if not Map.out_of_bound(self.location):
            return m.grids[self.location.y][self.location.x] == 1 # on platform
        else:
            return False


    # have valid path = have ways to advance to the next row
    # including walking sideways

    # return if valid path exist,
    # and x coordinate of a reachable platform on the next row

    # look for forward walk/jump first,
    def have_valid_path(self, m: Map) -> Status:
       pass


class Player(Entity):
    def __init__(self, coordinate, max_freeze=3, freezer_reset=7, redbull_reset=10):

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

        self.FREEZER_RESET = freezer_reset
        self.freezer_cooldown = self.FREEZER_RESET

        self.REDBULL_RESET = redbull_reset
        self.redbull_cooldown = self.REDBULL_RESET


    def walk(self, dir: Coordinate, m: Map) -> Status:
        s = super().walk(dir)
        m.expand(dir.y)
        return s

    def jump(self, dir: Coordinate, m: Map) -> Status:
        s = super().jump(dir)
        m.expand(dir.y*2)
        return s

    @property
    def has_freezer(self) -> bool:
        return self.freezer_cooldown == 0

    @property
    def has_redbull(self) -> bool:
        return self.redbull_cooldown == 0

    # freeze platforms
    def freezer(self, m: Map) -> Status:
        if not self.has_freezer:
            return Status(False, 0)

        self.freezer_cooldown = self.FREEZER_RESET # reset cooldown

        count = 0
        for freeze_coord in self.can_freeze:
            coord = self.location + freeze_coord
            if not Map.out_of_bound(coord): # only freeze what's possible
                if m.grids[coord.y][coord.x] == 0:
                    m.grids[coord.y][coord.x] = 1
                    count += 1

        # using freezer when it has no effect = failure
        return Status(bool(count), 0)


    def redbull(self, m: Map) -> Status:
        if not self.has_redbull:
            return Status(False, 0)

        self.redbull_cooldown = self.REDBULL_RESET # reset countdown

        # TODO: better way to select new platform
        x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)
        while m.grids[y][x] == 0 or (x, y) == (self.location.x, self.location.y): # lava | current pos
            x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)

        dy = y - self.location.y
        self.location = Coordinate(x=x, y=y)
        m.expand(dy)

        return Status(True, dy)


    # a player may walk, jump, use freezer,
    # freezer has cooldown
    # def valid_walk(self, map: Map) -> set:
    #    result = set()

    #    for action, c in enumerate(action_to_direction):
    #         target = self.location + c
    #         if Map.out_of_bound(target):
    #             if map.map[target.y][target.x] == 1:
    #                 result.add(action)


class Monster(Entity):

    _DIRECTIONS = (
                        Coordinate(x=0, y=1), # up, walk
                        Coordinate(x=0, y=-1), # down, walk
                        Coordinate(x=-1, y=0), # left, walk
                        Coordinate(x=1, y=0), # right, walk

                        Coordinate(x=0, y=2), # up, jump
                        Coordinate(x=0, y=-2), # down, jump
                        Coordinate(x=-2, y=0), # left, jump
                        Coordinate(x=2, y=0), # right, jump

                        Coordinate(x=-1, y=1), # up-left, walk
                        Coordinate(x=1, y=1), # up-right, walk
                        Coordinate(x=-1, y=-1), # down-left, walk
                        Coordinate(x=1, y=-1), # down-right, walk
                       )

    def __init__(self, coordinate = Coordinate(x=-1, y=-1)):
        Entity.__init__(self, coordinate)


    def respawn(self, player_location: Coordinate, m: Map) -> None:
        self.location = Coordinate(x=2, y=5)


    def step(self, p: Coordinate, m: Map) -> None:
        # kill if dropped into lava by player
        if not self.is_alive(m):
            self.location = OFF_SCREEN

        for d in self._DIRECTIONS:
            if (self.location + d) == p: # if directly catch player
                self.location += d
                return

        v = p - self.location
        best_actions = sorted(
            self._DIRECTIONS,
            key=lambda t: t.x*v.x + t.y*v.y,    # maximize dot product
            reverse=True)

        for a in best_actions:
            target = self.location + a
            if m.grids[target.y][target.x] != 0: # select action that get closest to player
                self.location = target
                return

        # otherwise, kill monster
        self.location = OFF_SCREEN


RenderState = namedtuple("RenderState", ["player_loc", "monster_loc", "freezer", "redbull", "slice"])

class Playground:

    # 0-7: walk (see action_to_direction)
    # 8-15: destroy
    # 16-19: jump (up down left right)
    # 20: freezer, 21: redbull

    def __init__(self, monster_respawn=3) -> None:
        self.map = Map()
        self.player = Player(Coordinate(x=MAP_WIDTH//2, y=MAP_HEIGHT//2))
        self.monster = Monster()

        # after monster died, takes a while until monster respawns
        self.MONSTER_RESPAWN = monster_respawn
        self.monster_respawn_cooldown = self.MONSTER_RESPAWN

        self.score = 0

        # to be removed (
        self._key_to_action = {
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
        self.is_jump = False
        self.is_destroy = False
        # )

        self._action_to_direction = (
                                    Coordinate(x=0, y=1), # up
                                    Coordinate(x=0, y=-1), # down
                                    Coordinate(x=-1, y=0), # left
                                    Coordinate(x=1, y=0), # right
                                    Coordinate(x=-1, y=1), # up-left
                                    Coordinate(x=1, y=1), # up-right
                                    Coordinate(x=-1, y=-1), # down-left
                                    Coordinate(x=1, y=-1), # down-right
                                )

    # to be removed
    def key_to_action(self, key: pygame.key) -> None:
        action = None

        try:
            key = self._key_to_action[pygame.key.name(key)]
            if key not in range(0, 12):
                raise KeyError

            if key == 8:
                self.is_jump = True
                self.is_destroy = False

            elif key == 9:
                action = 20 # freezer
                self.is_jump = False
                self.is_destroy = False

            elif key == 10:
                action = 21 # redbull
                self.is_jump = False
                self.is_destroy = False

            elif key == 11:
                self.is_jump = False
                self.is_destroy = True

            else:
                if self.is_jump and key in range(0, 4):
                    action = 16 + key # jump + dir
                    self.is_jump = False
                elif self.is_destroy:
                    action = 8 + key # destroy + dir
                    self.is_destroy = False
                elif not self.is_jump and not self.is_destroy:
                    action = key # walk + dir

        except KeyError:
            return

        return action

    @property
    def is_player_alive(self) -> bool:
        return self.player.is_alive(self.map) and \
               not (self.player.location == self.monster.location) # on platform & not caught by monster


    def play(self, action: int) -> Status:
        # player action
        # match case can't match range yet
        used_freezer = False
        used_redbull = False
        if action in range(0, 7+1):
            s = self.player.walk(self._action_to_direction[action], self.map)
        elif action in range(8, 15+1):
            s = self.player.destroy(self._action_to_direction[action-8], self.map)
        elif action in range(16, 19+1):
            s = self.player.jump(self._action_to_direction[action-16], self.map)
        elif action == 20:
            s = self.player.freezer(self.map)
            used_freezer = s.success # may be a spammed use
        elif action == 21:
            s = self.player.redbull(self.map)
            used_redbull = s.success
        else:
            raise ValueError("Unknown Action")


        if not used_freezer:
            if self.player.freezer_cooldown == 0:
                self.player.freezer_cooldown = 0 # capped at 0 if player did not use
            else: # countdown started
                self.player.freezer_cooldown -= 1

        if not used_redbull:
            if self.player.redbull_cooldown == 0:
                self.player.redbull_cooldown = 0
            else:
                self.player.redbull_cooldown -= 1


        # player kill himself
        if not self.is_player_alive:
            return Status(False, -10)

        # decrease spawn cooldown
        if Map.out_of_bound(self.monster.location): # not yet spawned
            self.monster_respawn_cooldown -= 1

        # if monster spawned in this round, don't step it
        if self.monster_respawn_cooldown == 0: # spawned new monster
            self.monster_respawn_cooldown = self.MONSTER_RESPAWN # reset cooldown
            self.monster.respawn(self.player.location, self.map)

        elif self.monster_respawn_cooldown == self.MONSTER_RESPAWN: # cooldown already reset, i.e., monster already spawned
            self.monster.step(self.player.location, self.map)

        # if player is caught by monster
        if not self.is_player_alive:
           return Status(False, -10)

        self.score += s.score
        print(f"Player: {self.player.location.coord(index=False)} | Monster: {self.monster.location.coord(index=False)}")
        print(f"Freezer cooldown: {self.player.freezer_cooldown} | Redbull cooldown: {self.player.redbull_cooldown}")
        if Map.out_of_bound(self.monster.location):
            print(f"Monster respawned in {self.monster_respawn_cooldown} rounds")
        print("----")
        return s

    def _get_slice(self):
        centre = self.player.location.y
        if centre < MAP_HEIGHT // 2:
            return self.map.grids[:MAP_HEIGHT][::-1]
        else:
            return self.map.grids[centre-MAP_HEIGHT//2: centre+MAP_HEIGHT//2+1][::-1]


    # used for rendering
    # output is in pygame coordinates
    @property
    def render_state(self):
        ploc, mloc = self.player.location, self.monster.location    # alias

        # player location
        if ploc.y < MAP_HEIGHT // 2:
            player_coord = Coordinate(x=ploc.x, y=MAP_HEIGHT - 1 - ploc.y)
        else:
            player_coord = Coordinate(x=ploc.x, y=MAP_HEIGHT//2)

        # monster location
        if ploc.y <= MAP_HEIGHT //2: # around start of the map
            if mloc.y <= MAP_HEIGHT - 1:
                monster_coord = Coordinate(x=mloc.x, y=(MAP_HEIGHT - 1) - mloc.y)
            else:
                monster_coord = OFF_SCREEN

        else:   # after walking far
            if abs(mloc.y - ploc.y) <= MAP_HEIGHT // 2: # must be around player
                monster_coord = Coordinate(
                    x=mloc.x,
                    y=(MAP_HEIGHT - 1) - (MAP_HEIGHT // 2 + (mloc.y - ploc.y))
                    ) # offset from centre
            else:
                monster_coord = OFF_SCREEN


        return RenderState(
            player_loc=player_coord,
            monster_loc=monster_coord,
            freezer=self.player.freezer_cooldown,
            redbull=self.player.redbull_cooldown,
            slice=self._get_slice(),
        )

    @staticmethod
    def _binary(l: list) -> int:
        result = 0
        for index, element in enumerate(reversed(l)):
            result += element * (2 ** index)
        return result

    # return 1d vector state specifically for RL algo
    # player xy, monster xy, freezer&redbull cooldown,
    # then $(MAP_HEIGHT) numbers representing rows
    @property
    def rl_state(self) -> tuple:
        s = self.render_state
        return (
            *s.player_loc.coord(index=False),
            *s.monster_loc.coord(index=False),
            s.freezer,
            s.redbull,
            *[self._binary(i) for i in s.slice]
        )


class Window:
    """
    handles pygame window rendering,
    including sprites, map, score display, etc.
    """

    font = pygame.font.get_fonts()

    def __init__(self, playground: Playground, fps:int|None) -> None:
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("The Floor is Lava")

        self.fps = fps

        self.GRID_SIZE = pygame.image.load("assets/lava.png").get_height()
        self.STAT_HEIGHT = 2

        self.win_size = (MAP_WIDTH * self.GRID_SIZE, (MAP_HEIGHT + self.STAT_HEIGHT) * self.GRID_SIZE) # resolution pending
        self.win = pygame.display.set_mode(self.win_size)

        self.game_surface = pygame.Surface((MAP_WIDTH * self.GRID_SIZE, MAP_HEIGHT * self.GRID_SIZE))
        self.stat_surface = pygame.Surface((MAP_WIDTH * self.GRID_SIZE, self.STAT_HEIGHT * self.GRID_SIZE))
        self.stat_surface.fill("darkorange2") # background colour

        self.lava_image = pygame.image.load("assets/lava.png").convert()
        self.platform_image = pygame.image.load("assets/platform.png").convert()
        self.platform_lip_image = pygame.image.load("assets/platform_lip.png").convert_alpha()
        self.player_image = pygame.image.load("assets/player.png").convert_alpha()
        self.monster_image = pygame.image.load("assets/monster.png").convert_alpha()

        self.clock = pygame.time.Clock()

        self.playground = playground


    # most of the rendering belongs to here
    def draw(self) -> None:

        if self.fps is not None:
            self.clock.tick(self.fps)

        s = self.playground.render_state

        # draw grids of lava / platform
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):

                topleft = (j*self.GRID_SIZE, i*self.GRID_SIZE)

                if s.slice[i][j] == 0: # lava
                    self.game_surface.blit(
                        self.lava_image,
                        self.lava_image.get_rect(topleft=topleft)
                        )
                    # draw front side of platform if there is platform above
                    if i-1 >= 0 and s.slice[i-1][j] == 1:
                        self.game_surface.blit(
                            self.platform_lip_image,
                            self.platform_lip_image.get_rect(topleft=topleft)
                            )
                else: # platform
                    self.game_surface.blit(
                        self.platform_image,
                        self.platform_image.get_rect(topleft=topleft)
                    )


        # offset grid size // 3 to center the sprite
        self.game_surface.blit(
            self.player_image,
            self.player_image.get_rect(
                center=(
                    (s.player_loc.x+0.5) * self.GRID_SIZE,
                    (s.player_loc.y+0.5) * self.GRID_SIZE - self.GRID_SIZE//3
                    )
                )
            )


        if not Map.out_of_bound(s.monster_loc):
            self.game_surface.blit(
                self.monster_image,
                self.monster_image.get_rect(
                    center=(
                        (s.monster_loc.x+0.5) * self.GRID_SIZE,
                        (s.monster_loc.y+0.5) * self.GRID_SIZE - self.GRID_SIZE//3
                        )
                    )
                )

        self.win.blit(
            self.game_surface,
            self.game_surface.get_rect(
                topleft=(0, self.STAT_HEIGHT * self.GRID_SIZE)
                )
            )
        self.win.blit(self.stat_surface, self.stat_surface.get_rect(topleft=(0, 0)))

        pygame.display.flip()


class MainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None, fps=None, trunc=300) -> None:

        # 1D vector:
        # player xy, monster xy, freezer&redbull cooldown,
        # then 15 numbers representing rows
        self.obs_space = gym.spaces.Box(shape=(6+MAP_HEIGHT,))

        # 8 walk, 8 destroy, 4 jump, freezer, redbull
        self.act_space = gym.spaces.Discrete(22)

        self.playground = Playground()

        # objects for rendering
        self.fps = fps
        self.window = Window(self.playground, self.fps)

        self.render_mode = render_mode
        self.trunc = trunc  # truncate after $(trunc) steps
        self.step_count = 0 # no. of steps taken, including invalid ones


    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)    # reset RNG

        self.step_count = 0
        self.playground = Playground()
        self.window = Window(self.playground, self.fps)

        observation = self.playground.rl_state()
        info = {
            "step_count": self.step_count,
            "score": 0
        }

        self._render_frame()

        return observation, info


    def step(self, action) -> tuple:

        self.step_count += 1
        status = self.playground.play(action)

        observation = self.playground.rl_state()

        if status.success is False and status.score == 0:
            reward = -1
        else:
            reward = status.score

        terminated = (status.score == -10)
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


if __name__ == "__main__":

    playground = Playground()
    win = Window(playground=playground, fps=15)

    running = True
    # s = Status(True, 0)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

            action = None
            if event.type == pygame.KEYDOWN:
                action = playground.key_to_action(event.key)

            if action is not None:
                playground.play(action)

        win.draw()

        if not playground.is_player_alive:
            running = False

    pygame.quit()

import pygame
import gym
import numpy as np
from itertools import product
from collections import namedtuple

#TODO:
# Redbull teleporation
# Initial platform generation
# New platform generation
# Difficulty control
# Text display
# Game sound


# keyboard mapping
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
    def __init__(self, map_width, map_height) -> None:
        # the map maybe better stored by lists rather than ndarray,
        # as the map grows as the player moves forward

        # we may hardcode the first few rows
        # and dynamically generate the rest
        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.grids = [[0 for i in range(self.MAP_WIDTH)] for j in range(self.MAP_HEIGHT)] # actual map

        # TODO: move this to playground
        # generate initial platforms
        self.init_platform_size = 5
        for j in range(self.MAP_HEIGHT//2 - self.init_platform_size//2, self.MAP_HEIGHT):
            for i in range(self.MAP_WIDTH//2 - self.init_platform_size//2, self.MAP_WIDTH//2 + self.init_platform_size//2 + 1):
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

    def out_of_bound(self, c: Coordinate) -> bool:
        return c.x < 0 or c.x >= self.MAP_WIDTH or c.y < 0
    
    def is_lava(self, c: Coordinate) -> bool:
        if not self.out_of_bound(c):
            return self.grids[c.y][c.x] == 0
        else:
            return False


# score = score given to player
# success + score determines reward
# e.g. even if score = 0, can still have -ve reward
Status = namedtuple("Status", ["success", "score"])
INVALID_STATUS = Status(False, 0)   # failed but no penalty
DEAD_STATUS = Status(False, -10)


class Entity:

    def __init__(self, start_loc: Coordinate):
        self.location = start_loc

    # move 1 block up/down/left/right
    # loc is passed by reference
    def walk(self, dir: Coordinate, m: Map) -> Status:
        newloc = self.location + dir

        if m.out_of_bound(newloc):
            return INVALID_STATUS
        else:
            self.location = newloc
            if m.is_lava(self.location):
                return DEAD_STATUS
            else:
                return Status(True, dir.y)

    # can jump 2 platforms to up/down/left/right, but not diagonal
    def jump(self, dir: Coordinate, m: Map) -> Status:
        newloc = self.location + dir * 2

        if m.out_of_bound(newloc):
            return INVALID_STATUS
        else:
            self.location = newloc
            if m.is_lava(self.location):
                return DEAD_STATUS
            else:
                return Status(True, dir.y * 2)


    def is_alive(self, m: map) -> bool:
        if not m.out_of_bound(self.location):
            return not m.is_lava(self.location) # on platform
        else:
            return False


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
        s = super().walk(dir, m)
        m.expand(dir.y)
        return s

    def jump(self, dir: Coordinate, m: Map) -> Status:
        s = super().jump(dir, m)
        m.expand(dir.y*2)
        return s
    
    def destroy(self, dir: Coordinate, m: Map) -> None:
        target = self.location + dir

        if m.out_of_bound(target):
            return INVALID_STATUS

        if m.grids[target.y][target.x] == 0:
            return INVALID_STATUS
        else:
            m.grids[target.y][target.x] = 0
            return Status(True, 0)

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

        for freeze_coord in self.can_freeze:
            coord = self.location + freeze_coord
            if not m.out_of_bound(coord): # only freeze what's possible
                if m.is_lava(coord):
                    m.grids[coord.y][coord.x] = 1

        return Status(True, 0)


    def redbull(self, m: Map) -> Status:
        if not self.has_redbull:
           return Status(False, 0)

        self.redbull_cooldown = self.REDBULL_RESET # reset countdown

        # TODO: better way to select new platform
        MAP_WIDTH = 9
        MAP_HEIGHT = 15
        x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0,MAP_HEIGHT-1)
        while m.grids[y][x] == 0 or (x, y) == (self.location.x, self.location.y): # lava | current pos
            x, y = np.random.randint(0, MAP_WIDTH-1), np.random.randint(0, MAP_HEIGHT-1)

        dy = y - self.location.y
        self.location = Coordinate(x=x, y=y)
        m.expand(dy)

        return Status(True, dy)


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
        # TODO: respawn monster around player
        self.location = Coordinate(x=2, y=5)


    def step(self, p: Coordinate, m: Map) -> None:
        # kill if dropped into lava by player
        if not self.is_alive(m):
            self.location = OFF_SCREEN
        
        v = p - self.location
        
        if v in self._DIRECTIONS:  # if directly catches player
            self.location = p
            return

        # find action closest to vector from monster to player
        # by maximizing dot product,
        # prefer actions that reach further (i.e. jump > walk)
        
        best_actions = sorted(
            self._DIRECTIONS,
            key=lambda t: t.x*v.x + t.y*v.y,
            reverse=True)

        for a in best_actions:
            target = self.location + a
            if not m.out_of_bound(target):
                if m.grids[target.y][target.x] != 0:
                    self.location = target
                    return
        
        # kill monster when no path available
        self.location = OFF_SCREEN


RenderState = namedtuple(
    "RenderState",
    ["player_loc", "monster_loc", "freezer", "redbull", "slice", "score"]
)

class Playground:

    def __init__(self, map_width, map_height, monster_respawn=3, play_mode = "human") -> None:
        self.play_mode = play_mode
        self.difficulty = None

        self.MAP_WIDTH = map_width # to be link with difficulty
        self.MAP_HEIGHT = map_height # to be link with difficulty
        self.map = Map(self.MAP_WIDTH, self.MAP_HEIGHT)

        self.player = Player(Coordinate(x=self.MAP_WIDTH//2, y=self.MAP_HEIGHT//2))
        self.monster = Monster()

        # after monster died, takes a while until monster respawns
        self.MONSTER_RESPAWN = monster_respawn # to be link with difficulty
        self.monster_respawn_cooldown = self.MONSTER_RESPAWN

        self.score = 0

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

    @property
    def is_monster_spawned(self) -> bool:
        return not self.map.out_of_bound(self.monster.location)


    def _set_tool_cooldown(self, used_freezer, used_redbull) -> None:
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


    def play(self, action: int) -> Status:
        # player action
        used_freezer = False
        used_redbull = False

        # 0-7: walk (see action_to_direction)
        # 8-15: destroy
        # 16-19: jump (up down left right)
        # 20: freezer, 21: redbull
        
        if action in range(0, 7+1):
            s = self.player.walk(self._action_to_direction[action], self.map)
        
        elif action in range(8, 15+1):
            s = self.player.destroy(self._action_to_direction[action-8], self.map)
        
        elif action in range(16, 19+1):
            s = self.player.jump(self._action_to_direction[action-16], self.map)
        
        elif action == 20:
            s = self.player.freezer(self.map)
            used_freezer = True
        
        elif action == 21:
            s = self.player.redbull(self.map)
            used_redbull = True
        
        else:
            raise ValueError("Unknown Action")

        # ignore no-ops
        # e.g. using tools before cooldown finished
        if s == INVALID_STATUS:
            return
        
        # don't decrement cooldown 
        # if just sucessfully used a tool this round
        # (as cooldown has just been reset)
        if not used_freezer and self.player.freezer_cooldown > 0:
            self.player.freezer_cooldown -= 1
            
        if not used_redbull and self.player.redbull_cooldown > 0:
            self.player.redbull_cooldown -= 1


        # if player suicided
        if not self.is_player_alive:
            return DEAD_STATUS

        # decrease spawn cooldown
        if not self.is_monster_spawned: # not yet spawned
            self.monster_respawn_cooldown -= 1

        # monster action: spawn monster or step if already spawned
        # if monster spawned in this round, don't step it
        if self.monster_respawn_cooldown == 0:
            self.monster_respawn_cooldown = self.MONSTER_RESPAWN
            self.monster.respawn(self.player.location, self.map)

        elif self.monster_respawn_cooldown == self.MONSTER_RESPAWN: # monster already spawned
            # self.monster.step(self.player.location, self.map)
            pass

        # if player is caught by monster
        if not self.is_player_alive:
           return DEAD_STATUS

        self.score += s.score

        print(f"Player: {self.player.location.coord(index=False)} | Monster: {self.monster.location.coord(index=False)}")
        print(f"Freezer cooldown: {self.player.freezer_cooldown} | Redbull cooldown: {self.player.redbull_cooldown}")
        if self.map.out_of_bound(self.monster.location):
            print(f"Monster respawned in {self.monster_respawn_cooldown} rounds")
        print("--------------------")

        return s

    def _get_slice(self):
        centre = self.player.location.y
        if centre < self.MAP_HEIGHT // 2:
            return self.map.grids[:self.MAP_HEIGHT][::-1]
        else:
            return self.map.grids[centre-self.MAP_HEIGHT//2: centre+self.MAP_HEIGHT//2+1][::-1]


    # used for rendering
    # output is in pygame coordinates
    @property
    def render_state(self):
        ploc, mloc = self.player.location, self.monster.location    # alias

        # player location
        if ploc.y < self.MAP_HEIGHT // 2:
            player_coord = Coordinate(x=ploc.x, y=self.MAP_HEIGHT - 1 - ploc.y)
        else:
            player_coord = Coordinate(x=ploc.x, y=self.MAP_HEIGHT//2)

        # monster location
        if ploc.y <= self.MAP_HEIGHT //2: # around start of the map
            if mloc.y <= self.MAP_HEIGHT - 1:
                monster_coord = Coordinate(x=mloc.x, y=(self.MAP_HEIGHT - 1) - mloc.y)
            else:
                monster_coord = OFF_SCREEN

        else:   # after walking far
            if abs(mloc.y - ploc.y) <= self.MAP_HEIGHT // 2: # must be around player
                monster_coord = Coordinate(
                    x=mloc.x,
                    y=(self.MAP_HEIGHT - 1) - (self.MAP_HEIGHT // 2 + (mloc.y - ploc.y))
                    ) # offset from centre
            else:
                monster_coord = OFF_SCREEN


        return RenderState(
            player_loc=player_coord,
            monster_loc=monster_coord,
            freezer=self.player.freezer_cooldown,
            redbull=self.player.redbull_cooldown,
            slice=self._get_slice(),
            score=self.score
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

    #font = pygame.font.get_fonts()

    def __init__(self, playground: Playground, fps:int|None) -> None:
        self.playground = playground
        self.MAP_WIDTH = self.playground.MAP_WIDTH
        self.MAP_HEIGHT = self.playground.MAP_HEIGHT

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("The Floor is Lava")
        pygame.font.init()

        self.fps = fps

        self.GRID_SIZE = pygame.image.load("assets/lava.png").get_height()
        self.STAT_HEIGHT = 2

        self.win_size = (self.MAP_WIDTH * self.GRID_SIZE, (self.MAP_HEIGHT + self.STAT_HEIGHT) * self.GRID_SIZE) # resolution pending
        self.win = pygame.display.set_mode(self.win_size)

        self.game_surface = pygame.Surface((self.MAP_WIDTH * self.GRID_SIZE, self.MAP_HEIGHT * self.GRID_SIZE))
        self.stat_surface = pygame.Surface((self.MAP_WIDTH * self.GRID_SIZE, self.STAT_HEIGHT * self.GRID_SIZE))

        self.font = pygame.font.Font('freesansbold.ttf', 16)

        self.lava_image = pygame.image.load("assets/lava.png").convert()
        self.platform_image = pygame.image.load("assets/platform.png").convert()
        self.platform_lip_image = pygame.image.load("assets/platform_lip.png").convert_alpha()
        self.player_image = pygame.image.load("assets/player.png").convert_alpha()
        self.monster_image = pygame.image.load("assets/monster.png").convert_alpha()

        self.clock = pygame.time.Clock()

    # drawing gameplay area
    def draw_game(self, s: RenderState) -> None:

        # draw grids of lava / platform
        for i in range(self.MAP_HEIGHT):
            for j in range(self.MAP_WIDTH):

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


        #if not Map.out_of_bound(s.monster_loc):
        if not s.monster_loc == OFF_SCREEN:
            self.game_surface.blit(
                self.monster_image,
                self.monster_image.get_rect(
                    center=(
                        (s.monster_loc.x+0.5) * self.GRID_SIZE,
                        (s.monster_loc.y+0.5) * self.GRID_SIZE - self.GRID_SIZE//3
                        )
                    )
                )

    # EXPERIMENTAL
    # drawing stat bar on top
    def draw_stat(self, s: RenderState) -> None:
        freezer_text = self.font.render(f"Freezer: {s.freezer} steps", True, "white")
        redbull_rext = self.font.render(f"Redbull: {s.redbull} steps", True, "white")
        score_text = self.font.render(f"{s.score}", True, "white")

        self.stat_surface.fill("darkorange1") # background colour
        self.stat_surface.blit(freezer_text, freezer_text.get_rect(topleft=(4, 4)))
        self.stat_surface.blit(redbull_rext, redbull_rext.get_rect(topleft=(4, 4 + self.GRID_SIZE)))
        self.stat_surface.blit(score_text, score_text.get_rect(topleft=((self.MAP_WIDTH - 2)*self.GRID_SIZE, 8 + self.GRID_SIZE)))


    # most of the rendering belongs to here
    def draw(self) -> None:

        if self.fps is not None:
            self.clock.tick(self.fps)

        s = self.playground.render_state
        self.draw_game(s)
        self.draw_stat(s)

        self.win.blit(
            self.game_surface,
            self.game_surface.get_rect(
                topleft=(0, self.STAT_HEIGHT * self.GRID_SIZE)
                )
            )

        self.win.blit(
            self.stat_surface,
            self.stat_surface.get_rect(
                topleft=(0, 0)
                )
            )

        pygame.display.flip()


class MainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, map_width=9, map_height=15, render_mode=None, fps=None, trunc=300) -> None:

        # 1D vector:
        # player xy, monster xy, freezer&redbull cooldown,
        # then 15 numbers representing rows
        self.obs_space = gym.spaces.Box(shape=(6+map_height,))

        # 8 walk, 8 destroy, 4 jump, freezer, redbull
        self.act_space = gym.spaces.Discrete(22)

        # actual game
        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.playground = Playground(self.MAP_WIDTH, self.MAP_HEIGHT, play_mode = self.render_mode)

        # objects for rendering
        self.fps = fps
        self.window = Window(self.playground, self.fps)

        self.render_mode = render_mode
        self.trunc = trunc  # truncate after $(trunc) steps
        self.step_count = 0 # no. of steps taken, including invalid ones


    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)    # reset RNG

        self.step_count = 0
        self.playground = Playground(self.MAP_WIDTH, self.MAP_HEIGHT, play_mode = self.render_mode)
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

    playground = Playground(map_width=9, map_height=15)
    win = Window(playground=playground, fps=15)

    running = True
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

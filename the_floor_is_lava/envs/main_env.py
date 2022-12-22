import pygame
import gym
import numpy as np
from itertools import product
from functools import reduce
from collections import namedtuple


keys_to_action = {
    ('w', '8'): 0, # walk; up
    ('s', '2'): 1, # walk; down
    ('a', '4'): 2, # walk; left
    ('d', '6'): 3, # walk; right
    ('q', '7'): 4, # walk; up-left
    ('e', '9'): 5, # walk; up-right
    ('z', '1'): 6, # walk; down-left
    ('c', '3'): 7, # walk; down-right

    ('kw', '.8'): 8, # destroy; up
    ('ks', '.2'): 9, # destroy; down
    ('ka', '.4'): 10, # destroy; left
    ('kd', '.6'): 11, # destroy; right
    ('kq', '.7'): 12, # destroy; up-left
    ('ke', '.9'): 13, # destroy; up-right
    ('kz', '.1'): 14, # destroy; down-left
    ('kc', '.3'): 15, # destroy; down-right

    (' w', '08'): 16, # jump; up
    (' s', '02'): 17, # jump; down
    (' a', '04'): 18, # jump; left
    (' d', '06'): 19, # jump; right

    ('f', '*'): 20, # freezer
    ('r', '-'): 21, # redbull

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
    def __init__(self, map_width: int, map_height: int) -> None:

        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.grids = [] # actual map

    def out_of_bound(self, c: Coordinate) -> bool:
        return c.x < 0 or c.x >= self.MAP_WIDTH or c.y < 0

    def is_lava(self, c: Coordinate) -> bool:
        if not self.out_of_bound(c):
            return self.grids[c.y][c.x] == 0
        else:
            return False

    # generate 3 rows at a time
    # r = avg % of platforms on entire map
    # a = P(a particular row being blank)
    # use player location as seed, deterministic
    def expand(self, r: int, a: int, seed: tuple) -> None:
        rng = np.random.default_rng(seed=seed)

        ac = 1 - a
        # avg % of platforms on a row
        p = r / (ac**3 + 2*a*(ac**2) + (a**2)*ac)

        rows = np.zeros((3, self.MAP_WIDTH))
        for i in range(3):
            if rng.random() > a:
                rows[i] = rng.choice(2, size=self.MAP_WIDTH, p=(1-p, p))

        self.grids += rows.tolist()



# score = score given to player
# success + score determines reward
# e.g. even if score = 0, can still have -ve reward
Status = namedtuple("Status", ["success", "score"])
INVALID_STATUS = Status(False, 0)   # failed but no penalty
DEAD_STATUS = Status(False, -10)


class Entity:

    DIRECTIONS = (
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
    def __init__(self, coordinate: Coordinate, freezer_reset: int, redbull_reset: int):

        Entity.__init__(self, coordinate)

        self.max_freeze = 3

        _iter1= range(-self.max_freeze, self.max_freeze+1)
        _iter2 = range(-self.max_freeze, self.max_freeze+1)

        # manhattan distance
        self.can_freeze = [
            Coordinate(x=i, y=j)
            for i, j in product(_iter1, _iter2)
            if abs(i)+abs(j) <= self.max_freeze
            and not (i == 0 and j == 0)
            ]

        self.FREEZER_RESET = freezer_reset
        self.freezer_cooldown = self.FREEZER_RESET

        self.REDBULL_RESET = redbull_reset
        self.redbull_cooldown = self.REDBULL_RESET


    def destroy(self, dir: Coordinate, m: Map) -> None:
        target = self.location + dir

        if m.out_of_bound(target):
            return INVALID_STATUS

        if m.is_lava(target):
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
           return INVALID_STATUS

        self.freezer_cooldown = self.FREEZER_RESET # reset cooldown

        for freeze_coord in self.can_freeze:
            coord = self.location + freeze_coord
            if not m.out_of_bound(coord): # only freeze what's possible
                if m.is_lava(coord):
                    m.grids[coord.y][coord.x] = 1

        return Status(True, 0)


    def redbull(self, m: Map, max_range: int) -> Status:
        if not self.has_redbull:
           return INVALID_STATUS

        self.redbull_cooldown = self.REDBULL_RESET # reset countdown

        # teleport player randomly forward max_range//2 to max_range blocks, can be sideways
        x, dy = np.random.randint(0, len(m.grids[0])-1), np.random.randint(max_range//2, max_range)

        while m.is_lava(Coordinate(x=x, y=self.location.y + dy)):
            x, dy = np.random.randint(0, len(m.grids[0])-1), np.random.randint(max_range//2, max_range)

        self.location = Coordinate(x=x, y=self.location.y + dy)

        return Status(True, dy)


class Monster(Entity):

    def __init__(self, coordinate: Coordinate = Coordinate(x=-1, y=-1)):
        Entity.__init__(self, coordinate)


    def respawn(self, player_location: Coordinate, m: Map) -> None:

        rng = np.random.default_rng(seed=player_location.coord(index=False))

        # spawn as low as possible
        for i in range(-m.MAP_HEIGHT//2+1, m.MAP_HEIGHT//2 + 1):
            spawn_y = i + player_location.y

            if not m.out_of_bound(Coordinate(x=0, y=spawn_y)):
                possible_x = list(filter(
                    lambda x: not m.is_lava(Coordinate(x=x, y=spawn_y)),
                    range(m.MAP_WIDTH)
                ))

                if possible_x:
                    spawn_x = rng.choice(possible_x)
                    self.location = Coordinate(x=spawn_x, y=spawn_y)
                    break


    def step(self, p: Coordinate, m: Map) -> None:
        # kill if dropped into lava by player
        if not self.is_alive(m):
            self.location = OFF_SCREEN

        v = p - self.location

        if v in self.DIRECTIONS:  # if directly catches player
            self.location = p
            return

        # find action closest to vector from monster to player
        # by maximizing dot product,
        # prefer actions that reach further (i.e. jump > walk)

        best_actions = sorted(
            self.DIRECTIONS,
            key=lambda t: t.x*v.x + t.y*v.y,
            reverse=True)

        for a in best_actions:
            target = self.location + a
            if not m.out_of_bound(target):
                if not m.is_lava(target):
                    self.location = target
                    return

        # kill monster when no path available
        self.location = OFF_SCREEN


RenderState = namedtuple(
    "RenderState",
    ["player_loc", "monster_loc", "slice", "score", "freezer", "redbull", "monster_respawn",]
)

class Playground:
    _difficulty_to_var = {
        0: {"initial_platform_size": 7, "r": 0.45, "a": 0.15, "monster_respawn": 5, "freezer_reset": 5, "redbull_reset": 7,},
        1: {"initial_platform_size": 5, "r": 0.4, "a": 0.2, "monster_respawn": 3, "freezer_reset": 7, "redbull_reset": 10,},
        2: {"initial_platform_size": 5, "r": 0.3, "a": 0.25, "monster_respawn": 1, "freezer_reset": 7, "redbull_reset": 15,}
    }
    # 0: Easy (very easy)
    # 1: Normal (current)
    # 2: Hard (very difficult, may require tools at start or even consecutively)


    def __init__(self, map_width: int, map_height: int, difficulty: int):
        self.difficulty = difficulty

        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.map = Map(self.MAP_WIDTH, self.MAP_HEIGHT)

        # size of initial platforms at centre
        self.init_platform_size = self._difficulty_to_var[self.difficulty]["initial_platform_size"]
        # percentage of platforms in each row
        self.p_perc_platform = self._difficulty_to_var[self.difficulty]["r"]
        # percentage of totally blank rows per 3 rows generated
        self.p_blank_row = self._difficulty_to_var[self.difficulty]["a"]

        centre = Coordinate(x=self.MAP_WIDTH//2, y=self.MAP_HEIGHT//2)

        for i in range(self.MAP_HEIGHT // 3 + 1): # fill the starting grids with random platforms
            self.map.expand(
                            r=self.p_perc_platform,
                            a=0,
                            seed=(np.random.randint(centre.x - self.init_platform_size // 2, centre.x + self.init_platform_size // 2), \
                                  np.random.randint(centre.y - self.init_platform_size // 2, centre.y + self.init_platform_size // 2))
                                 )

        # spawn platforms at centre
        for j in range(centre.y - self.init_platform_size//2, centre.y + self.init_platform_size//2 + 1):
            for i in range(centre.x - self.init_platform_size//2, centre.x + self.init_platform_size//2 + 1):
                self.map.grids[j][i] = 1


        self.player = Player(
                             coordinate=centre,
                             freezer_reset=self._difficulty_to_var[self.difficulty]["freezer_reset"],
                             redbull_reset=self._difficulty_to_var[self.difficulty]["redbull_reset"]
                            ) # player always spawn at centre

        self.monster = Monster()

        if self.difficulty != 0: # Easy: Monster will not spawn at start
            self.monster.respawn(self.player.location, self.map)

        # after monster died, takes a while until monster respawns
        self.MONSTER_RESPAWN = self._difficulty_to_var[self.difficulty]["monster_respawn"]
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


        pygame.mixer.init()
        # background music
            # pygame.mixer.music.load("")
            # pygame.mixer.music.play(loops=-1) # infinite loop

        self.sounds = {
            "player_walk": pygame.mixer.Sound("assets/footstep.wav"),
            "player_jump": pygame.mixer.Sound("assets/footstep.wav"),
            "player_destroy": pygame.mixer.Sound("assets/destroy.wav"),
            "player_freezer": pygame.mixer.Sound("assets/freezer.flac"),
            "freezer_reset": pygame.mixer.Sound("assets/freezer_reset.wav"),
            "player_redbull": pygame.mixer.Sound("assets/redbull.wav"),
            "redbull_reset": pygame.mixer.Sound("assets/redbull_reset.wav"),
            "player_die": pygame.mixer.Sound("assets/player_die.wav"),
            "monster_attack": pygame.mixer.Sound("assets/monster_attack.mp3"),
            "monster_respawn": pygame.mixer.Sound("assets/monster_respawn.wav"),
        }

        for s in self.sounds:
            self.sounds[s].set_volume(0.5)


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
                                'f': 9, # freezer
                                'r': 10, # redbull
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

    @property
    def map_exhausted(self) -> bool:
        return (len(self.map.grids) - self.player.location.y) <= self.map.MAP_HEIGHT // 2

    def play(self, action: int) -> tuple[Status, str]:
        # player action
        used_freezer = False
        used_redbull = False

        # 0-7: walk (up down left right up-left up-right down-left down-right)
        # 8-15: destroy
        # 16-19: jump (up down left right)
        # 20: freezer, 21: redbull

        if action in range(0, 7+1):
            s = self.player.walk(self._action_to_direction[action], self.map)
            if not s == INVALID_STATUS:
                self.sounds["player_walk"].play()

        elif action in range(8, 15+1):
            s = self.player.destroy(self._action_to_direction[action-8], self.map)
            if not s == INVALID_STATUS:
                self.sounds["player_destroy"].play()

        elif action in range(16, 19+1):
            s = self.player.jump(self._action_to_direction[action-16], self.map)
            if not s == INVALID_STATUS:
                self.sounds["player_jump"].play()

        elif action == 20:
            s = self.player.freezer(self.map)
            if not s == INVALID_STATUS:
                used_freezer = True
                self.sounds["player_freezer"].play()
            else:
                return s, "freezer_reset"

        elif action == 21:
            s = self.player.redbull(self.map, self.MAP_HEIGHT // 2)
            if not s == INVALID_STATUS:
                self.monster.location = OFF_SCREEN
                self.monster_respawn_cooldown = 1 # immediate monster respawn
                used_redbull = True
                self.sounds["player_redbull"].play()
            else:
                return s, "redbull_reset"

        else:
            raise ValueError("Unknown Action")

        # ignore no-ops
        # e.g. using tools before cooldown finished
        if s == INVALID_STATUS:
            return s, None

        # expand map
        while self.map_exhausted:
            self.map.expand(
                r=self.p_perc_platform,
                a=self.p_blank_row,
                seed=self.player.location.coord(index=False)
            )


        # don't decrement cooldown
        # if just sucessfully used a tool this round
        # (as cooldown has just been reset)
        if not used_freezer and self.player.freezer_cooldown > 0:
            self.player.freezer_cooldown -= 1
            if self.player.freezer_cooldown == 0:
                self.sounds["freezer_reset"].play()

        if not used_redbull and self.player.redbull_cooldown > 0:
            self.player.redbull_cooldown -= 1
            if self.player.redbull_cooldown == 0:
                self.sounds["redbull_reset"].play()


        # if player suicided
        if not self.is_player_alive:
            self.sounds["player_die"].play()
            return DEAD_STATUS, "player_death1"


        # decrease monster spawn cooldown
        if not self.is_monster_spawned: # not yet spawned
            self.monster_respawn_cooldown -= 1

        # monster action: spawn monster or step if already spawned
        # if monster spawned in this round, don't step it
        if self.monster_respawn_cooldown == 0:
            self.monster_respawn_cooldown = self.MONSTER_RESPAWN
            self.monster.respawn(self.player.location, self.map)
            self.sounds["monster_respawn"].play()

        elif self.monster_respawn_cooldown == self.MONSTER_RESPAWN: # monster already spawned
            self.monster.step(self.player.location, self.map)

        # if player is caught by monster
        if not self.is_player_alive:
            self.sounds["monster_attack"].play()
            return DEAD_STATUS, "player_death2"

        # update player's score
        self.score += s.score

        # kill monster if it is too far away from player
        if self.player.location.y - self.monster.location.y > round(self.map.MAP_HEIGHT * 0.7) :
            self.monster.location = OFF_SCREEN

        print(f"Player: {self.player.location.coord(index=False)} | Monster: {self.monster.location.coord(index=False)}")

        if not self.monster.is_alive(self.map):
            return s, "monster_respawn"

        return s, None


    def _get_slice(self) -> list[list[int]]:
        centre = self.player.location.y
        if centre < self.MAP_HEIGHT // 2:
            return self.map.grids[:self.MAP_HEIGHT][::-1]
        else:
            return self.map.grids[centre-self.MAP_HEIGHT//2: centre+self.MAP_HEIGHT//2+1][::-1]


    # used for rendering
    # output is in pygame coordinates
    @property
    def render_state(self) -> RenderState:
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
            slice=self._get_slice(),
            score=self.score,
            freezer=self.player.freezer_cooldown,
            redbull=self.player.redbull_cooldown,
            monster_respawn=self.monster_respawn_cooldown,
        )


    # return 1d vector state specifically for RL algo
    # player xy, monster xy, freezer&redbull cooldown,
    # then binary representation of rows
    @property
    def rl_state(self) -> tuple:
        s = self.render_state
        return (
            *s.player_loc.coord(index=False),
            *s.monster_loc.coord(index=False),
            s.freezer,
            s.redbull,
            *[reduce(lambda a,b: 2*a + b, i) for i in s.slice]
        )


class Window:
    """
    handles pygame window rendering,
    including sprites, map, score display, etc.
    """

    def __init__(self, playground: Playground, fps:int|None):
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
        self.msg_surface = pygame.Surface((self.MAP_WIDTH * self.GRID_SIZE, self.MAP_HEIGHT // 2 * self.GRID_SIZE))
        self.msg_surface.set_colorkey("gray")

        self.lava_image = pygame.image.load("assets/lava.png").convert()
        self.platform_image = pygame.image.load("assets/platform.png").convert()
        self.platform_lip_image = pygame.image.load("assets/platform_lip.png").convert_alpha()
        self.player_image = pygame.image.load("assets/player.png").convert_alpha()
        self.monster_image = pygame.image.load("assets/monster.png").convert_alpha()

        self.font = pygame.font.Font('assets/font.ttf', 16)
        self.font.set_bold(True)
        self.score_font = pygame.font.Font('assets/font.ttf', 32)
        self.msg_font = pygame.font.Font('assets/font.ttf', 14)
        self.msg_font.set_bold(True)

        self._msg_countdown = {
            "freezer_reset": {"DURATION": 2000, "countdown": 2000,},
            "redbull_reset": {"DURATION": 2000, "countdown": 2000,},
            "monster_respawn": {"DURATION": 2000, "countdown": 2000,},
            "player_death1": {"DURATION": 5000, "countdown": 5000,},
            "player_death2": {"DURATION": 5000, "countdown": 5000,},
        }

        self._msg_queue = []

        self.clock = pygame.time.Clock()


    # draw gameplay area
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


    # drawing statistics bar on top
    def draw_stat(self, s: RenderState) -> None:

        freezer_text = self.font.render(f"Freezer: {s.freezer} steps", False, "white")
        redbull_rext = self.font.render(f"Redbull: {s.redbull} steps", False, "white")
        score_text = self.score_font.render(f"{s.score}", False, "white")

        # draw lava background
        for i in range(self.STAT_HEIGHT):
            for j in range(self.MAP_WIDTH):
                self.stat_surface.blit(
                    self.lava_image,
                    self.lava_image.get_rect(
                        topleft=(j*self.GRID_SIZE, i*self.GRID_SIZE)
                        )
                )

        self.stat_surface.blit(
            freezer_text,
            freezer_text.get_rect(
                topleft=(self.GRID_SIZE // 4, self.GRID_SIZE // 4)
                )
            )

        self.stat_surface.blit(
            redbull_rext,
            redbull_rext.get_rect(
                topleft=(self.GRID_SIZE // 4, self.GRID_SIZE // 4 + self.GRID_SIZE)
                )
            )

        self.stat_surface.blit(
            score_text,
            score_text.get_rect(
                top=self.GRID_SIZE // 2, right=(self.MAP_WIDTH - 0.5) * self.GRID_SIZE
                )
            )


    # show game messages on screen, if any
    def print_msg(self, msg: str, s: RenderState, dt: int) -> None:

        # turn game message into text shown on screen
        def msg_to_text() -> pygame.font.Font:
            if msg == "freezer_reset":
                return self.msg_font.render(f"Freezer resets in {s.freezer} steps", False, "white")

            if msg == "redbull_reset":
                return self.msg_font.render(f"Redbull resets in {s.redbull} steps", False, "white")

            if msg == "monster_respawn":
                return self.msg_font.render(f"Monster respawns in {s.monster_respawn} steps", False, "white")

            if msg == "player_death1":
                return self.msg_font.render(f"You fell in lava!", False, "white")

            if msg == "player_death2":
                return self.msg_font.render(f"You are caught by monster!", False, "white")


        # append a new not-yet-blitted message
        if msg is not None and msg not in self._msg_queue:
            if msg == "player_death1" or msg == "player_death2":
                self._msg_queue.clear()

            self._msg_queue.append(msg)


        # remove the relevant message if:
        # monster has already respawned
        if "monster_respawn" in self._msg_queue and msg != "monster_respawn" and not s.monster_loc == OFF_SCREEN:
            self._msg_countdown["monster_respawn"]["countdown"] = self._msg_countdown["monster_respawn"]["DURATION"]
            self._msg_queue.remove("monster_respawn")

        # freezer already reset
        if "freezer_reset" in self._msg_queue and s.freezer == 0:
            self._msg_countdown["freezer_reset"]["countdown"] = self._msg_countdown["freezer_reset"]["DURATION"]
            self._msg_queue.remove("freezer_reset")

        # redbull already reset
        if "redbull_reset" in self._msg_queue and s.redbull == 0:
            self._msg_countdown["redbull_reset"]["countdown"] = self._msg_countdown["redbull_reset"]["DURATION"]
            self._msg_queue.remove("redbull_reset")


        # decrease the countdown of each message, or remove it if countdown is over
        for msg in self._msg_queue:
            if self._msg_countdown[msg]["countdown"] > 0:
                self._msg_countdown[msg]["countdown"] -= dt
            else:
                self._msg_countdown[msg]["countdown"] = self._msg_countdown[msg]["DURATION"]
                self._msg_queue.pop(0)


        self.msg_surface.fill("gray")
        for i, msg in enumerate(self._msg_queue):
            text = msg_to_text()
            self.msg_surface.blit(
                text,
                text.get_rect(top=i * self.GRID_SIZE, centerx=(self.MAP_WIDTH * self.GRID_SIZE)// 2)
            )


    # most of the rendering belongs to here
    def draw(self, msg: str = None) -> None:

        if self.fps is not None:
            dt = self.clock.tick(self.fps)

        s = self.playground.render_state
        self.draw_game(s)
        self.draw_stat(s)
        self.print_msg(msg, s, dt)

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

        self.win.blit(
            self.msg_surface,
            self.msg_surface.get_rect(
                topleft=(0, (self.STAT_HEIGHT + self.MAP_HEIGHT // 2 + 2) * self.GRID_SIZE)
                )
            )

        pygame.display.flip()


class MainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, map_width=9, map_height=15, difficulty=1, render_mode=None, fps=None, trunc=300) -> None:

        # 1D vector:
        # player xy, monster xy, freezer&redbull cooldown,
        # then 15 numbers representing rows
        self.obs_space = gym.spaces.Box(shape=(6+map_height,))

        # 8 walk, 8 destroy, 4 jump, freezer, redbull
        self.act_space = gym.spaces.Discrete(22)

        # actual game
        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.difficulty = difficulty
        self.playground = Playground(self.MAP_WIDTH, self.MAP_HEIGHT, self.difficulty)

        # objects for rendering
        self.fps = fps
        self.window = Window(self.playground, self.fps)

        self.render_mode = render_mode
        self.trunc = trunc  # truncate after $(trunc) steps
        self.step_count = 0 # no. of steps taken, including invalid ones


    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)    # reset RNG

        self.step_count = 0
        self.playground = Playground(self.MAP_WIDTH, self.MAP_HEIGHT, self.difficulty)
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
        status, game_msg = self.playground.play(action)

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

        self._render_frame(game_msg)

        return observation, reward, terminated, truncated, info


    def render(self) -> None:
        # outputting frames for training isn't required
        return None


    def _render_frame(self, game_msg: str = None) -> None:
        if self.render_mode == "human":
            self.window.draw(game_msg)


    def close(self) -> None:
        if self.window is not None:
            #pygame.mixer.music.stop()
            pygame.mixer.quit()
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":

    playground = Playground(map_width=9, map_height=15, difficulty=1)
    win = Window(playground=playground, fps=15)

    running = True
    while running:
        msg = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

            action = None
            if event.type == pygame.KEYDOWN:
                action = playground.key_to_action(event.key)

            if action is not None:
                s, msg = playground.play(action)

        win.draw(msg)

        if not playground.is_player_alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

    #pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.display.quit()
    pygame.quit()

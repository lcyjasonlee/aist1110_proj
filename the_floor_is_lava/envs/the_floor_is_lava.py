import pygame
import numpy as np
from itertools import product
from functools import reduce
from collections import namedtuple
from .coord import Coordinate


OFF_SCREEN = Coordinate(x=-1, y=-1)


class Map:
    """
    handles map creation and validation of coordinates in map
    """
    def __init__(self, map_width: int, map_height: int) -> None:

        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.grids = [] # actual map

    def out_of_bound(self, c: Coordinate) -> bool:
        '''
        check if the coordinate is out of bound
        '''

        return c.x < 0 or c.x >= self.MAP_WIDTH or c.y < 0

    def is_lava(self, c: Coordinate) -> bool:
        '''
        check if the coordinate is lava
        '''

        if not self.out_of_bound(c):
            return self.grids[c.y][c.x] == 0
        else:
            return False

    # generate 3 rows at a time
    # r = avg % of platforms on entire map
    # a = P(a particular row being blank)
    # use player location as seed
    def expand(self, r: int, a: int, seed: int|tuple|None = None) -> None:
        '''
        expand the map by generating 3 rows at a time
        '''

        rng = np.random.default_rng(seed=seed)

        ac = 1 - a
        # avg % of platforms on a row
        p = r / (ac**3 + 2*a*(ac**2) + (a**2)*ac)

        rows = np.zeros((3, self.MAP_WIDTH))
        for i in range(3):
            if rng.random() > a:
                rows[i] = rng.choice(2, size=self.MAP_WIDTH, p=(1-p, p))

        self.grids += rows.tolist()

    def reset(self):
        self.grids = []


# score = score given to player
# success + score determines reward
# e.g. even if score = 0, can still have -ve reward
Status = namedtuple("Status", ["success", "score"])
INVALID_STATUS = Status(False, 0)   # failed but no penalty
DEAD_STATUS = Status(False, -10)


class Entity:
    '''
    handles entity action (walk, jump),
    also validate entity location
    '''

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
        self.start_loc = start_loc
        self.location = start_loc


    def walk(self, dir: Coordinate, m: Map) -> Status:
        '''
        walk the entity by 1 block
        '''

        newloc = self.location + dir

        if m.out_of_bound(newloc):
            return INVALID_STATUS
        else:
            self.location = newloc
            if m.is_lava(self.location):
                return DEAD_STATUS
            else:
                return Status(True, dir.y)


    def jump(self, dir: Coordinate, m: Map) -> Status:
        '''
        jump the entity by 2 blocks, up/down/left/right
        '''

        newloc = self.location + dir * 2

        if m.out_of_bound(newloc):
            return INVALID_STATUS
        else:
            self.location = newloc
            if m.is_lava(self.location):
                return DEAD_STATUS
            else:
                return Status(True, dir.y * 2)


    def in_lava(self, m: map) -> bool:
        '''
        check if the entity is in lava
        '''

        if not m.out_of_bound(self.location):
            return m.is_lava(self.location)
        else:
            return False

    def reset(self) -> None:
        self.location = self.start_loc

class Player(Entity):
    '''
    handles player action (walk, jump, destroy) and tools (freezer, redbull)
    '''

    def __init__(self, coordinate: Coordinate, freezer_reset: int, redbull_reset: int):

        Entity.__init__(self, coordinate)

        # freeze range = 3, manhattan distance
        self.can_freeze = [
            Coordinate(x=i, y=j)
            for i, j in product(range(-3, 3+1), range(-3, 3+1))
            if abs(i)+abs(j) <= 3
            and not (i == 0 and j == 0)
            ]

        self.FREEZER_RESET = freezer_reset
        self.freezer_cooldown = 0 # tools available at start

        self.REDBULL_RESET = redbull_reset
        self.redbull_cooldown = 0


    def destroy(self, dir: Coordinate, m: Map) -> None:
        '''
        destroy the specified platform
        '''

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
        '''
        freeze nearby lava
        '''

        if not self.has_freezer:
           return INVALID_STATUS

        self.freezer_cooldown = self.FREEZER_RESET # reset cooldown

        for freeze_coord in self.can_freeze:
            coord = self.location + freeze_coord
            if not m.out_of_bound(coord): # only freeze what's possible
                if m.is_lava(coord):
                    m.grids[coord.y][coord.x] = 1

        return Status(True, 0)

    def redbull(self, m: Map) -> Status:
        '''
        teleport player forward 5-7 blocks randomly
        '''

        if not self.has_redbull:
           return INVALID_STATUS

        self.redbull_cooldown = self.REDBULL_RESET

        rng = np.random.default_rng(seed=self.location.coord(index=False))
        x, dy = -1, -1

        targets = list(product(range(m.MAP_WIDTH), range(5, 7+1)))
        targets = np.array(targets)
        
        rng = np.random.default_rng(self.location.coord(index=False))
        rng.shuffle(targets)
            
        for loc in targets:
            x, dy = loc
            if not m.is_lava(Coordinate(x=x, y=self.location.y + dy)):
                self.location = Coordinate(x=x, y=self.location.y + dy)
                return Status(True, dy)
        
        return Status(True, 0)

    def reset(self) -> None:
        super().reset()
        self.freezer_cooldown = 0
        self.redbull_cooldown = 0
        

class Monster(Entity):
    '''
    handles monster action (walk, jump) and respawning
    '''

    def __init__(self, coordinate: Coordinate = Coordinate(x=-1, y=-1)):
        Entity.__init__(self, coordinate)


    def respawn(self, player_location: Coordinate, m: Map) -> None:
        '''
        respawn the monster
        '''

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
        '''
        monster action
        '''

        # kill if dropped into lava by player
        if self.in_lava(m):
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
    ["player_loc", "monster_loc", "slice", "score", "freezer", "redbull"]
)

Difficulty = namedtuple(
    "Difficulty",
    ["init_platform_size", "r", "a", "respawn", "freezer_reset", "redbull_reset"]
)


class Actions:
    '''
    map each action to an ID value
    '''

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7

    # modifiers
    MOD_DESTROY = 8
    MOD_JUMP = 16

    FREEZER = 20
    REDBULL = 21

    WALK_SET = {UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}

    JUMP_SET = {MOD_JUMP+UP, MOD_JUMP+DOWN, MOD_JUMP+LEFT, MOD_JUMP+RIGHT}
    JUMPABLE_SET = {UP, DOWN, LEFT, RIGHT}

    DESTROY_SET = {MOD_DESTROY+UP, MOD_DESTROY+DOWN, MOD_DESTROY+LEFT,
                   MOD_DESTROY+RIGHT, MOD_DESTROY+UP_LEFT, MOD_DESTROY+UP_RIGHT,
                   MOD_DESTROY+DOWN_LEFT, MOD_DESTROY+DOWN_RIGHT}

    TOOL_SET = {FREEZER, REDBULL}


class Events:
    '''
    map each game event to an ID value
    '''

    PLAYER_WALK = 0
    PLAYER_JUMP = 1
    PLAYER_DESTROY = 2

    PLAYER_FREEZER = 3
    FREEZER_RESET = 4
    PLAYER_REDBULL = 5
    REDBULL_RESET = 6

    MONSTER_ATTACK = 7
    MONSTER_RESPAWN = 8

    PLAYER_DEATH = 9
    FELL_IN_LAVA = PLAYER_DEATH + 1
    WALK_TO_MONSTER = PLAYER_DEATH + 2
    CAUGHT_BY_MONSTER = PLAYER_DEATH + 3

    DEATH_SET = {PLAYER_DEATH, FELL_IN_LAVA, WALK_TO_MONSTER, CAUGHT_BY_MONSTER}


class Playground(Actions, Events):
    '''
    handles main game play,
    including map generation, and player and monster actions,
    also returns states for rendering and training
    '''

    # generate starting platform
    @staticmethod
    def _map_init(m: Map, size: int, r: float, seed: int) -> None:
        m.reset()
        centre = Coordinate(x=m.MAP_WIDTH//2, y=m.MAP_HEIGHT//2)
        
        while len(m.grids) < m.MAP_HEIGHT: # fill the starting grids with random platforms
            m.expand(r=r, a=0, seed=seed)

        # spawn platforms at centre
        for j in range(centre.y - size//2, centre.y + size//2 + 1):
            for i in range(centre.x - size//2, centre.x + size//2 + 1):
                m.grids[j][i] = 1
    
    
    def __init__(self, map_width: int, map_height: int, difficulty: int, seed: int=None):
        # game settings

        self.difficulty = difficulty

        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.map = Map(self.MAP_WIDTH, self.MAP_HEIGHT)
        self.seed = seed
        

        self._difficulty_to_var = [
            Difficulty(7, 0.45, 0.15, -1, 5, 7),    # peaceful
            Difficulty(7, 0.45, 0.15, 5, 5, 7),
            Difficulty(5, 0.4, 0.2, 3, 7, 10),
            Difficulty(5, 0.3, 0.25, 1, 7, 15),
        ]

        (
            self.init_platform_size, # size of initial platforms at centre
            self.mapgen_r, # parameters for map generation
            self.mapgen_a,
            self.MONSTER_RESPAWN,
            self.freezer_reset,
            self.redbull_reset
        ) = self._difficulty_to_var[self.difficulty]


        # map generation
        self._map_init(self.map, self.init_platform_size, self.mapgen_r, seed)

        # entity initialization
        centre = Coordinate(x=self.MAP_WIDTH//2, y=self.MAP_HEIGHT//2)
        self.player = Player(
            coordinate=centre,
            freezer_reset=self.freezer_reset,
            redbull_reset=self.redbull_reset
        ) # player always spawn at centre

        self.monster = Monster()

        if self.difficulty > 1: # Easy: Monster will not spawn at start
            self.monster.respawn(self.player.location, self.map)

        # after monster died, takes a while until monster respawns
        self.monster_respawn_cooldown = self.MONSTER_RESPAWN


        # others
        self.score = 0

        self._action_to_direction = {
            Actions.UP         : Coordinate(x=0, y=1), # up
            Actions.DOWN       : Coordinate(x=0, y=-1), # down
            Actions.LEFT       : Coordinate(x=-1, y=0), # left
            Actions.RIGHT      : Coordinate(x=1, y=0), # right
            Actions.UP_LEFT    : Coordinate(x=-1, y=1), # up-left
            Actions.UP_RIGHT   : Coordinate(x=1, y=1), # up-right
            Actions.DOWN_LEFT  : Coordinate(x=-1, y=-1), # down-left
            Actions.DOWN_RIGHT : Coordinate(x=1, y=-1), # down-right
        }


    @property
    def is_player_alive(self) -> bool:
        return not self.player.in_lava(self.map) and \
               not (self.player.location == self.monster.location) # on platform & not caught by monster

    @property
    def is_monster_spawned(self) -> bool:
        return not self.map.out_of_bound(self.monster.location)

    @property
    def map_exhausted(self) -> bool:
        return (len(self.map.grids) - self.player.location.y) <= self.map.MAP_HEIGHT // 2


    def play(self, action: int) -> tuple[Status, list[str]]:
        '''
        carry out actions,
        return action status (for handling reward),
        and lists of events (for handling sounds & on screen msg)
        '''

        events = []

        # player action
        used_freezer = False
        used_redbull = False

        if action in Actions.WALK_SET:
            s = self.player.walk(self._action_to_direction[action], self.map)
            if not s == INVALID_STATUS:
                events.append(Events.PLAYER_WALK)

        elif action in Actions.DESTROY_SET:
            s = self.player.destroy(self._action_to_direction[action-8], self.map)
            if not s == INVALID_STATUS:
                events.append(Events.PLAYER_DESTROY)

        elif action in Actions.JUMP_SET:
            s = self.player.jump(self._action_to_direction[action-16], self.map)
            if not s == INVALID_STATUS:
                events.append(Events.PLAYER_JUMP)

        elif action == Actions.FREEZER:
            s = self.player.freezer(self.map)
            if not s == INVALID_STATUS:
                used_freezer = True
                events.append(Events.PLAYER_FREEZER)

        elif action == Actions.REDBULL:
            s = self.player.redbull(self.map)
            if not s == INVALID_STATUS:
                self.monster.location = OFF_SCREEN
                self.monster_respawn_cooldown = 1 # immediate monster respawn
                used_redbull = True
                events.append(Events.PLAYER_REDBULL)

        else:
            raise ValueError("Unknown Action")

        # ignore no-ops
        # e.g. using tools before cooldown finished
        if s == INVALID_STATUS:
            return s, events

        # expand map
        while self.map_exhausted:
            self.map.expand(
                r=self.mapgen_r,
                a=self.mapgen_a,
                seed=self.player.location.coord(index=False)
            )

        # don't decrement cooldown
        # if just sucessfully used a tool this round
        # (as cooldown has just been reset)
        if not used_freezer and self.player.freezer_cooldown > 0:
            self.player.freezer_cooldown -= 1
            if self.player.freezer_cooldown == 0:
                events.append(Events.FREEZER_RESET)

        if not used_redbull and self.player.redbull_cooldown > 0:
            self.player.redbull_cooldown -= 1
            if self.player.redbull_cooldown == 0:
                events.append(Events.REDBULL_RESET)


        # if player suicided:
        # either by falling into lava or walking to monster
        if not self.is_player_alive:
            events.append(Events.PLAYER_DEATH)

            if self.player.in_lava(self.map):
                events.append(Events.FELL_IN_LAVA)
            else:
                events.append(Events.MONSTER_ATTACK)
                events.append(Events.WALK_TO_MONSTER)

            return DEAD_STATUS, events

        # only step monster if not peaceful
        if self.MONSTER_RESPAWN > 0:
            
            # decrease monster spawn cooldown
            if not self.is_monster_spawned: # not yet spawned
                self.monster_respawn_cooldown -= 1

            # monster action: spawn monster or step if already spawned
            # if monster spawned in this round, don't step it
            if self.monster_respawn_cooldown == 0:
                self.monster_respawn_cooldown = self.MONSTER_RESPAWN
                self.monster.respawn(self.player.location, self.map)
                events.append(Events.MONSTER_RESPAWN)

            elif self.monster_respawn_cooldown == self.MONSTER_RESPAWN: # monster already spawned
                self.monster.step(self.player.location, self.map)


        # if player is caught by monster
        if not self.is_player_alive:
            events.append(Events.PLAYER_DEATH)
            events.append(Events.MONSTER_ATTACK)
            events.append(Events.CAUGHT_BY_MONSTER)
            return DEAD_STATUS, events


        # update player's score
        self.score += s.score

        # kill monster if it is too far away from player
        if self.player.location.y - self.monster.location.y > self.map.MAP_HEIGHT // 2 :
            self.monster.location = OFF_SCREEN

        return s, events


    def _get_slice(self) -> list[list[int]]:
        '''
        return slice of map for rendering
        '''

        centre = self.player.location.y
        if centre < self.MAP_HEIGHT // 2:
            return self.map.grids[:self.MAP_HEIGHT][::-1]
        else:
            return self.map.grids[centre-self.MAP_HEIGHT//2: centre+self.MAP_HEIGHT//2+1][::-1]


    @property
    def render_state(self) -> RenderState:
        '''
        return render state for rendering
        (in pygame coordinates)
        '''

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
            redbull=self.player.redbull_cooldown
        )


    @property
    def rl_state(self) -> tuple:
        '''
        return 1d vector state specifically for RL algo
        player xy, monster xy, freezer & redbull cooldown,
        and binary representation of rows
        '''

        s = self.render_state
        
        return np.concatenate(
            (
                *s.player_loc.coord(index=False),
                *s.monster_loc.coord(index=False),
                s.freezer,
                s.redbull,
                *[reduce(lambda a,b: 2*a + b, i) for i in s.slice]
            ),
            dtype=np.float32
        )

    def reset(self) -> None:
        self._map_init(self.map, self.init_platform_size, self.mapgen_r, self.seed)
        self.player.reset()
        self.monster.reset()
        if self.difficulty > 1:
            self.monster.respawn(self.player.location, self.map)
        self.score = 0

class Window:
    """
    handles pygame window rendering,
    including map, entities, and score and tool countdown display
    """

    def __init__(self, playground: Playground, fps: int|None):
        self.playground = playground
        self.MAP_WIDTH = self.playground.MAP_WIDTH
        self.MAP_HEIGHT = self.playground.MAP_HEIGHT

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("The Floor is Lava")

        self.GRID_SIZE = pygame.image.load("assets/lava.png").get_height()
        self.STAT_HEIGHT = 2

        self.win_size = (self.MAP_WIDTH * self.GRID_SIZE, (self.MAP_HEIGHT + self.STAT_HEIGHT) * self.GRID_SIZE)
        self.win = pygame.display.set_mode(self.win_size)

        self.game_surface = pygame.Surface((self.MAP_WIDTH * self.GRID_SIZE, self.MAP_HEIGHT * self.GRID_SIZE))
        self.stat_surface = pygame.Surface((self.MAP_WIDTH * self.GRID_SIZE, self.STAT_HEIGHT * self.GRID_SIZE))

        self.lava_image = pygame.image.load("assets/lava.png").convert()
        self.platform_image = pygame.image.load("assets/platform.png").convert()
        self.platform_lip_image = pygame.image.load("assets/platform_lip.png").convert_alpha()
        self.player_image = pygame.image.load("assets/player.png").convert_alpha()
        self.monster_image = pygame.image.load("assets/monster.png").convert_alpha()

        self.stat_image = pygame.image.load("assets/stat.png").convert()
        self.freezer_image = pygame.image.load("assets/freezer.png").convert_alpha()
        self.freezer_bw_image = pygame.image.load("assets/freezer_bw.png").convert_alpha()
        self.redbull_image = pygame.image.load("assets/redbull.png").convert_alpha()
        self.redbull_bw_image = pygame.image.load("assets/redbull_bw.png").convert_alpha()


        pygame.font.init()
        self.FONT_FILE = 'assets/font.ttf'
        self.font = pygame.font.Font(self.FONT_FILE, 16)
        self.font.set_bold(True)

        self.score_font = pygame.font.Font(self.FONT_FILE, 24)

        self.fps = fps
        self.clock = pygame.time.Clock()


    def _draw_game(self, s: RenderState) -> None:
        '''
        draw game play area,
        including map and player and monster
        '''

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


    @staticmethod
    def _align_text(text: pygame.Surface, img_topleft: tuple, img_height: int) -> pygame.Rect:
        '''
        put text on bottomleft corner
        dynamically calculate size & position
        '''

        rect = text.get_rect()
        rect.height *= 0.8  # remove extra space below text
        rect.bottomleft = (img_topleft[0], img_topleft[1]+img_height)
        return rect


    def _draw_stat(self, s: RenderState) -> None:
        '''
        draw score and tool countdown bar on top
        '''

        texture_surface = pygame.Surface(self.stat_image.get_size())
        texture_surface.blit(self.stat_image, (0, 0))

        # draw score
        score_surface = self.score_font.render(f"{s.score:>4}", True, (255, 255, 0))
        score_loc = (3.1*self.GRID_SIZE, 0.74*self.GRID_SIZE)
        texture_surface.blit(score_surface, score_loc)


        # draw freezer & redbull
        freezer_loc = (5.47*self.GRID_SIZE, 0.5*self.GRID_SIZE)
        redbull_loc = (freezer_loc[0] + 2*self.GRID_SIZE, freezer_loc[1])

        if s.freezer == 0:
            texture_surface.blit(self.freezer_image, freezer_loc)
        else:
            texture_surface.blit(self.freezer_bw_image, freezer_loc)

            fcool_text = self.font.render(f"{s.freezer}", True, (255, 255, 0))
            fcool_rect = self._align_text(fcool_text, freezer_loc, self.freezer_image.get_height())

            # translucent black background for better text visibility
            fcool_bg = pygame.Surface(fcool_rect.size)
            fcool_bg.set_alpha(150)
            fcool_bg.fill((0,0,0))

            texture_surface.blit(fcool_bg, fcool_rect.topleft)
            texture_surface.blit(fcool_text, fcool_rect.topleft)


        if s.redbull == 0:
            texture_surface.blit(self.redbull_image, redbull_loc)
        else:
            texture_surface.blit(self.redbull_bw_image, redbull_loc)

            rcool_text = self.font.render(f"{s.redbull}", True, (255, 255, 0))
            rcool_rect = self._align_text(rcool_text, redbull_loc, self.redbull_image.get_height())

            rcool_bg = pygame.Surface(rcool_rect.size)
            rcool_bg.set_alpha(150)
            rcool_bg.fill((0,0,0))

            texture_surface.blit(rcool_bg, rcool_rect.topleft)
            texture_surface.blit(rcool_text, rcool_rect.topleft)


        # rendering is based on pre-baked texture,
        # align image to center of stat surface
        self.stat_surface.fill((35, 23, 9))
        self.stat_surface.blit(
            texture_surface,
            texture_surface.get_rect(
                midtop=self.stat_surface.get_rect().midtop
            )
        )


    def draw(self) -> None:
        '''
        rendering to screen
        '''

        if self.fps is not None:
            self.clock.tick(self.fps)

        s = self.playground.render_state

        self._draw_game(s)
        self.win.blit(
            self.game_surface,
            self.game_surface.get_rect(
                topleft=(0, self.STAT_HEIGHT * self.GRID_SIZE)
                )
            )

        self._draw_stat(s)
        self.win.blit(
            self.stat_surface,
            self.stat_surface.get_rect(
                topleft=(0, 0)
                )
            )

        pygame.display.flip()


    def direct_draw(self, surface: pygame.Surface, topleft_ratio: tuple[int, int] = (0,0)):
        '''
        directly blit a surface to the window,
        for external use (extending game capabilities)
        '''
        # auto calculate proportions:
        # (0,0)=topleft of screen, (1,1) = bottomright

        self.win.blit(
            surface,
            (
                topleft_ratio[0]*self.MAP_WIDTH*self.GRID_SIZE,
                topleft_ratio[1]*self.MAP_HEIGHT*self.GRID_SIZE
            )
        )
        pygame.display.flip()

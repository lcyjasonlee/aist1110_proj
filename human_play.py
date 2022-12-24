# adds main loop & additional pausing/game over screen

from the_floor_is_lava.envs.main_env import *
from cmdargs import args
import pygame

from platform import platform
from os import environ

if "WSL2" in platform():
    environ["SDL_AUDIODRIVER"] = "pulseaudio"
    

playground = Playground(
    map_width=args.mapwidth,
    map_height=args.mapheight,
    difficulty=args.difficulty,
    seed=args.seed
)

win = Window(
    playground=playground,
    fps=args.fps
)


# pause game until keypress detected or quit game
# return True if continue playing
def hold() -> bool:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.KEYDOWN:
                return True


def help_screen(w: Window) -> bool:
    help_image = pygame.image.load("assets/help.png")
    help_surface = pygame.Surface(w.win.get_size())
    
    # lava texture in background
    for row in range(w.win.get_height()):
        for col in range(w.win.get_width()):
            help_surface.blit(
                w.lava_image,
                (col*w.GRID_SIZE, row*w.GRID_SIZE)
            )
    
    # darken the lava
    darken_surface = pygame.Surface(w.win.get_size())
    darken_surface.set_alpha(160)
    darken_surface.fill((0,0,0))
    
    # place help image in the center
    help_surface.blit(darken_surface, (0,0))
    help_surface.blit(
        help_image, 
        help_image.get_rect(
            center=help_surface.get_rect().center
        )
    )

    w.direct_draw(help_surface, (0,0))
    return hold()
    

def death_screen(w: Window, events: list) -> bool:
    if Events.CAUGHT_BY_MONSTER in events:
        msg = "you got caught by the monster"
    elif Events.WALK_TO_MONSTER in events:
        msg = "the monster doesn't like your hugs"
    elif Events.FELL_IN_LAVA in events:
        msg = "you tried swimming in lava"
    else:
        return
    
    
    surface = pygame.Surface((w.MAP_WIDTH * w.GRID_SIZE, 4 * w.GRID_SIZE))
    surface.set_colorkey("gray")
    surface.fill("gray")
    
    msg_font = pygame.font.Font(w.FONT_FILE, 16)
    msg_text = msg_font.render(msg, False, "white")
    
    gameover_font = pygame.font.Font(w.FONT_FILE, 36)
    gameover_font.set_bold(True)
    gameover_text = gameover_font.render("YOU DIED!", False, "yellow")
    
    restart_font = pygame.font.Font(w.FONT_FILE, 16)
    restart_text = restart_font.render("PRESS ANY KEY TO RESTART", False, "green")
    
    surface.blit(
            gameover_text,
            gameover_text.get_rect(centerx=(surface.get_width() // 2))
    )
    
    surface.blit(
        msg_text,
        msg_text.get_rect(
            centerx=(surface.get_width() // 2),
            top = (gameover_text.get_height())
        )
    )
    
    surface.blit(
        restart_text,
        restart_text.get_rect(
            centerx=(surface.get_width() // 2),
            top = (gameover_text.get_height() + msg_text.get_height()) * 1.4
        )
    )
    
    bg = pygame.Surface(w.win.get_rect().size)
    bg.set_alpha(120)
    bg.fill((0,0,0))
    
    w.direct_draw(bg, (0,0))
    w.direct_draw(surface, (0, 0.5))
    return hold()
 

class Keys:
    pygame.key.set_repeat(200) # delay for continuous key presses

    _key_to_action_keyboard = {
        pygame.K_w: Actions.UP,
        pygame.K_s: Actions.DOWN,
        pygame.K_a: Actions.LEFT,
        pygame.K_d: Actions.RIGHT,
        pygame.K_q: Actions.UP_LEFT,
        pygame.K_e: Actions.UP_RIGHT,
        pygame.K_z: Actions.DOWN_LEFT,
        pygame.K_c: Actions.DOWN_RIGHT,

        pygame.K_v: Actions.MOD_DESTROY,
        pygame.K_SPACE: Actions.MOD_JUMP,
        pygame.K_f: Actions.FREEZER,
        pygame.K_r: Actions.REDBULL,
    }

    _key_to_action_numpad = {
        pygame.K_KP8: Actions.UP,
        pygame.K_KP2: Actions.DOWN,
        pygame.K_KP4: Actions.LEFT,
        pygame.K_KP6: Actions.RIGHT,
        pygame.K_KP7: Actions.UP_LEFT,
        pygame.K_KP9: Actions.UP_RIGHT,
        pygame.K_KP1: Actions.DOWN_LEFT,
        pygame.K_KP3: Actions.DOWN_RIGHT,

        pygame.K_KP_PERIOD: Actions.MOD_DESTROY,
        pygame.K_KP0: Actions.MOD_JUMP,
        pygame.K_KP_MULTIPLY: Actions.FREEZER,
        pygame.K_KP_MINUS: Actions.REDBULL,
    }

    def __init__(self):
        self._keyboard_jump = False
        self._numpad_jump = False
        self._keyboard_destroy = False
        self._numpad_destroy = False


    def combined_keys_check(self, key: pygame.key) -> None:
        if key == pygame.K_SPACE:
            self._keyboard_jump = False

        if key == pygame.K_KP0:
            self._numpad_jump = False

        if key == pygame.K_v:
            self._keyboard_destroy = False

        if key == pygame.K_KP_PERIOD:
            self._numpad_destroy = False


    # checking combined keys in >1 frames, exactly-simutaneous press might fail
    def key_to_action(self, key: pygame.key) -> int:
        if key not in self._key_to_action_keyboard and key not in self._key_to_action_numpad:
            return -1

        if key in self._key_to_action_keyboard:
            action = self._key_to_action_keyboard[key]

            if action == Actions.MOD_DESTROY:
                self._keyboard_destroy = True
                return -1

            elif action == Actions.MOD_JUMP:
                self._keyboard_jump = True
                return -1

            elif self._keyboard_destroy:
                if action in Actions.WALK_SET:
                    action += Actions.MOD_DESTROY
                else:
                    return -1

            elif self._keyboard_jump:
                if action in Actions.JUMPABLE_SET:
                    action += Actions.MOD_JUMP
                else:
                    return -1

            return action

        if key in self._key_to_action_numpad:
            action = self._key_to_action_numpad[key]

            if action == Actions.MOD_DESTROY:
                self._numpad_destroy = True
                return -1

            elif action == Actions.MOD_JUMP:
                self._numpad_jump = True
                return -1

            elif self._numpad_destroy:
                if action in Actions.WALK_SET:
                    action += Actions.MOD_DESTROY
                else:
                    return -1

            elif self._numpad_jump:
                if action in Actions.JUMPABLE_SET:
                    action += Actions.MOD_JUMP
                else:
                    return -1

            return action


    def _key_to_direction_keyboard2(self, keys) -> int:
        if keys[pygame.K_w]: return Actions.UP
        if keys[pygame.K_s]: return Actions.DOWN
        if keys[pygame.K_a]: return Actions.LEFT
        if keys[pygame.K_d]: return Actions.RIGHT
        if keys[pygame.K_q]: return Actions.UP_LEFT
        if keys[pygame.K_e]: return Actions.UP_RIGHT
        if keys[pygame.K_z]: return Actions.DOWN_LEFT
        if keys[pygame.K_c]: return Actions.DOWN_RIGHT
        return -1

    def _key_to_direction_numpad2(self, keys) -> int:
        if keys[pygame.K_KP8]: return Actions.UP
        if keys[pygame.K_KP2]: return Actions.DOWN
        if keys[pygame.K_KP4]: return Actions.LEFT
        if keys[pygame.K_KP6]: return Actions.RIGHT
        if keys[pygame.K_KP7]: return Actions.UP_LEFT
        if keys[pygame.K_KP9]: return Actions.UP_RIGHT
        if keys[pygame.K_KP1]: return Actions.DOWN_LEFT
        if keys[pygame.K_KP3]: return Actions.DOWN_RIGHT
        return -1

    # actually checking all keys pressed down at the moment, but highly unreliable (esp. jumping)
    def key_to_action2(self, keys) -> int:
        action = -1

        if keys[pygame.K_SPACE] or keys[pygame.K_0]:
            if keys[pygame.K_SPACE]:
                dir = self._key_to_direction_keyboard2(keys)
            else:
                dir = self._key_to_direction_numpad2(keys)

            if dir in Actions.JUMPABLE_SET:
                action = Actions.MOD_JUMP + dir

        elif keys[pygame.K_v] or keys[pygame.K_KP_PERIOD]:
            if keys[pygame.K_v]:
                dir = self._key_to_direction_keyboard2(keys)
            else:
                dir = self._key_to_direction_numpad2(keys)

            if dir != -1:
                action = Actions.MOD_DESTROY + dir

        elif keys[pygame.K_f] or keys[pygame.K_KP_MULTIPLY]:
            action = Actions.FREEZER

        elif keys[pygame.K_r] or keys[pygame.K_KP_MINUS]:
            action = Actions.REDBULL

        else:
            action = self._key_to_direction_keyboard2(keys)
            if action == -1:
                action = self._key_to_direction_numpad2(keys)

        return action


key = Keys()

pygame.mixer.init()
sounds = {
    Events.PLAYER_WALK: pygame.mixer.Sound("assets/footstep.wav"),
    Events.PLAYER_JUMP: pygame.mixer.Sound("assets/footstep.wav"),
    Events.PLAYER_DESTROY: pygame.mixer.Sound("assets/destroy.wav"),
    Events.PLAYER_FREEZER: pygame.mixer.Sound("assets/freezer.flac"),
    Events.FREEZER_RESET: pygame.mixer.Sound("assets/freezer_reset.wav"),
    Events.PLAYER_REDBULL: pygame.mixer.Sound("assets/redbull.wav"),
    Events.REDBULL_RESET: pygame.mixer.Sound("assets/redbull_reset.wav"),
    Events.PLAYER_DEATH: pygame.mixer.Sound("assets/player_die.wav"),
    Events.MONSTER_ATTACK: pygame.mixer.Sound("assets/monster_attack.mp3"),
    Events.MONSTER_RESPAWN: pygame.mixer.Sound("assets/monster_respawn.wav"),
}

for s in sounds:
    sounds[s].set_volume(0.5)


# starting screen
running = help_screen(win)

# main loop of game

while running:
    playground_events = []
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_SLASH, pygame.K_KP_DIVIDE):
            running = help_screen(win)

        if event.type == pygame.KEYUP:
            key.combined_keys_check(event.key)

        action = -1
        if event.type == pygame.KEYDOWN:
            action = key.key_to_action(event.key)
            #action = key.key_to_action2(pygame.key.get_pressed())

        if action != -1:
            s, playground_events = playground.play(action)
        
    for e in playground_events:
        if e in sounds:
            sounds[e].play()

    win.draw()

    if not playground.is_player_alive:
        running = death_screen(win, playground_events)
        
        playground = Playground(
            map_width=args.mapwidth,
            map_height=args.mapheight,
            difficulty=args.difficulty
        )

        win.playground = playground

pygame.quit()
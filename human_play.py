# intercept the playground class,
# adds main loop & additional pausing/game over screen

from the_floor_is_lava.envs.main_env import *
from cmdargs import args
import pygame


playground = Playground(
    map_width=args.mapwidth,
    map_height=args.mapheight,
    difficulty=args.difficulty
)

win = Window(
    playground=playground,
    fps=args.fps
)


def help_screen(w: Window):
    help_image = pygame.image.load("assets/help.png")
    w.win.blit(help_image, (0,0))
    pygame.display.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                running = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit(0)


class Keys:
    pygame.key.set_repeat(200) # delay for continuous key presses

    _key_to_action_keyboard = {
        pygame.K_w: UP,
        pygame.K_s: DOWN,
        pygame.K_a: LEFT,
        pygame.K_d: RIGHT,
        pygame.K_q: UP_LEFT,
        pygame.K_e: UP_RIGHT,
        pygame.K_z: DOWN_LEFT,
        pygame.K_c: DOWN_RIGHT,

        pygame.K_v: DESTROY,
        pygame.K_SPACE: JUMP,
        pygame.K_f: FREEZER,
        pygame.K_r: REDBULL,
    }

    _key_to_action_numpad = {
        pygame.K_KP8: UP,
        pygame.K_KP2: DOWN,
        pygame.K_KP4: LEFT,
        pygame.K_KP6: RIGHT,
        pygame.K_KP7: UP_LEFT,
        pygame.K_KP9: UP_RIGHT,
        pygame.K_KP1: DOWN_LEFT,
        pygame.K_KP3: DOWN_RIGHT,

        pygame.K_KP_PERIOD: DESTROY,
        pygame.K_KP0: JUMP,
        pygame.K_KP_MULTIPLY: FREEZER,
        pygame.K_KP_MINUS: REDBULL,
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

            if action == DESTROY:
                self._keyboard_destroy = True
                return -1

            elif action == JUMP:
                self._keyboard_jump = True
                return -1

            elif self._keyboard_destroy:
                if action in {UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}:
                    action += DESTROY
                else:
                    return -1

            elif self._keyboard_jump:
                if action in {UP, DOWN, LEFT, RIGHT}:
                    action += JUMP
                else:
                    return -1

            return action

        if key in self._key_to_action_numpad:
            action = self._key_to_action_numpad[key]

            if action == DESTROY:
                self._numpad_destroy = True
                return -1

            elif action == JUMP:
                self._numpad_jump = True
                return -1

            elif self._numpad_destroy:
                if action in {UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}:
                    action += DESTROY
                else:
                    return -1

            elif self._numpad_jump:
                if action in {UP, DOWN, LEFT, RIGHT}:
                    action += JUMP
                else:
                    return -1

            return action


    def _key_to_direction_keyboard2(self, keys) -> int:
        if keys[pygame.K_w]: return UP
        if keys[pygame.K_s]: return DOWN
        if keys[pygame.K_a]: return LEFT
        if keys[pygame.K_d]: return RIGHT
        if keys[pygame.K_q]: return UP_LEFT
        if keys[pygame.K_e]: return UP_RIGHT
        if keys[pygame.K_z]: return DOWN_LEFT
        if keys[pygame.K_c]: return DOWN_RIGHT
        return -1

    def _key_to_direction_numpad2(self, keys) -> int:
        if keys[pygame.K_KP8]: return UP
        if keys[pygame.K_KP2]: return DOWN
        if keys[pygame.K_KP4]: return LEFT
        if keys[pygame.K_KP6]: return RIGHT
        if keys[pygame.K_KP7]: return UP_LEFT
        if keys[pygame.K_KP9]: return UP_RIGHT
        if keys[pygame.K_KP1]: return DOWN_LEFT
        if keys[pygame.K_KP3]: return DOWN_RIGHT
        return -1

    # actually checking all keys pressed down at the moment, but highly unreliable (esp. jumping)
    def key_to_action2(self, keys) -> int:
        action = -1

        if keys[pygame.K_SPACE] or keys[pygame.K_0]:
            if keys[pygame.K_SPACE]:
                dir = self._key_to_direction_keyboard2(keys)
            else:
                dir = self._key_to_direction_numpad2(keys)

            if dir in {UP, DOWN, LEFT, RIGHT}:
                action = JUMP + dir

        elif keys[pygame.K_v] or keys[pygame.K_KP_PERIOD]:
            if keys[pygame.K_v]:
                dir = self._key_to_direction_keyboard2(keys)
            else:
                dir = self._key_to_direction_numpad2(keys)

            if dir != -1:
                action = DESTROY + dir

        elif keys[pygame.K_f] or keys[pygame.K_KP_MULTIPLY]:
            action = FREEZER

        elif keys[pygame.K_r] or keys[pygame.K_KP_MINUS]:
            action = REDBULL

        else:
            action = self._key_to_direction_keyboard2(keys)
            if action == -1:
                action = self._key_to_direction_numpad2(keys)

        return action


key = Keys()

pygame.mixer.init()
sounds = {
    PLAYER_WALK: pygame.mixer.Sound("assets/footstep.wav"),
    PLAYER_JUMP: pygame.mixer.Sound("assets/footstep.wav"),
    PLAYER_DESTROY: pygame.mixer.Sound("assets/destroy.wav"),
    PLAYER_FREEZER: pygame.mixer.Sound("assets/freezer.flac"),
    FREEZER_RESET: pygame.mixer.Sound("assets/freezer_reset.wav"),
    PLAYER_REDBULL: pygame.mixer.Sound("assets/redbull.wav"),
    REDBULL_RESET: pygame.mixer.Sound("assets/redbull_reset.wav"),
    PLAYER_DIE: pygame.mixer.Sound("assets/player_die.wav"),
    MONSTER_ATTACK: pygame.mixer.Sound("assets/monster_attack.mp3"),
    MONSTER_RESPAWN: pygame.mixer.Sound("assets/monster_respawn.wav"),
}

for s in sounds:
    sounds[s].set_volume(0.5)


# starting screen
help_screen(win)

end_msg = None

# main loop of game
running = True
while running:
    sound_queue = []

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

        if event.type == pygame.KEYUP:
            key.combined_keys_check(event.key)

        action = -1
        if event.type == pygame.KEYDOWN:
            action = key.key_to_action(event.key)
            #action = key.key_to_action2(pygame.key.get_pressed())

        if action != -1:
            s, sound_queue, end_msg = playground.play(action)

    for s in sound_queue:
        sounds[s].play()

    win.draw()

    if not playground.is_player_alive:
        running = False


# ending screen
running = True
while running:
    win.draw(end_msg)  # to be changed

    # close window only when player quits
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

pygame.quit()

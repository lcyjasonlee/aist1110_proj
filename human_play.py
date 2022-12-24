# adds main loop & additional pausing/game over screen

from the_floor_is_lava.envs.the_floor_is_lava import *
from cmdargs import args
from keys import Keys
import pygame

from platform import platform
from os import environ
import keys

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


def hold() -> bool:
    '''
    pause game until keypress detected or quit game,
    return True if continue playing
    '''

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.KEYDOWN:
                return True


def help_screen(w: Window) -> bool:
    '''
    show help screen at game start
    '''

    help_image = pygame.image.load("assets/help.png")
    help_surface = pygame.Surface(w.win.get_size())

    # lava texture in background
    for row in range(w.win.get_height() // w.GRID_SIZE):
        for col in range(w.win.get_width() // w.GRID_SIZE):
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
    '''
    show end screen after death
    '''

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


# sounds during gameplay
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
            running = help_screen(win) # helping screen at start

        if event.type == pygame.KEYUP:
            Keys.combined_keys_check(event.key)

        action = -1
        if event.type == pygame.KEYDOWN:
            action = Keys.key_to_action(event.key) # get action from key input

        if action != -1:
            s, playground_events = playground.play(action) # perform the action in game

    for e in playground_events:
        if e in sounds:
            sounds[e].play() # play sounds, if any

    win.draw()

    if not playground.is_player_alive:
        running = death_screen(win, playground_events) # show death screen

        # reset game
        playground = Playground(
            map_width=args.mapwidth,
            map_height=args.mapheight,
            difficulty=args.difficulty
        )

        win.playground = playground

pygame.quit()

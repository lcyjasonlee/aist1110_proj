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

win = Window(playground=playground, fps=args.fps)


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
        
help_screen(win)

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
            s = playground.play(action)

    win.draw(msg)

    if not playground.is_player_alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

#pygame.mixer.music.stop()
pygame.quit()
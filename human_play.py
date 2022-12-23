# intercept the playground class,
# adds main loop & additional pausing/game over screen

from the_floor_is_lava.envs.main_env import *

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
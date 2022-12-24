import pygame
from the_floor_is_lava.envs.main_env import Actions

class Keys:
    pygame.init()
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

    keyboard_jump = False
    numpad_jump = False
    keyboard_destroy = False
    numpad_destroy = False

    @staticmethod
    def combined_keys_check(key: pygame.key) -> None:
        if key == pygame.K_SPACE:
            Keys.keyboard_jump = False

        if key == pygame.K_KP0:
            Keys.numpad_jump = False

        if key == pygame.K_v:
            Keys.keyboard_destroy = False

        if key == pygame.K_KP_PERIOD:
            Keys.numpad_destroy = False


    # checking combined keys in >1 frames, exactly-simutaneous press might fail
    @staticmethod
    def key_to_action(key: pygame.key) -> int:
        if key not in Keys._key_to_action_keyboard and key not in Keys._key_to_action_numpad:
            return -1

        if key in Keys._key_to_action_keyboard:
            action = Keys._key_to_action_keyboard[key]

            if action == Actions.MOD_DESTROY:
                Keys.keyboard_destroy = True
                return -1

            elif action == Actions.MOD_JUMP:
                Keys.keyboard_jump = True
                return -1

            elif Keys.keyboard_destroy:
                if action + Actions.MOD_DESTROY in Actions.DESTROY_SET:
                    action += Actions.DESTROY
                else:
                    return -1

            elif Keys.keyboard_jump:
                if action in Actions.JUMPABLE_SET:
                    action += Actions.MOD_JUMP
                else:
                    return -1

            return action

        if key in Keys._key_to_action_numpad:
            action = Keys._key_to_action_numpad[key]

            if action == Actions.MOD_DESTROY:
                Keys.numpad_destroy = True
                return -1

            elif action == Actions.MOD_JUMP:
                Keys.numpad_jump = True
                return -1

            elif Keys.numpad_destroy:
                if action + Actions.MOD_DESTROY in Actions.DESTROY_SET:
                    action += Actions.MOD_DESTROY
                else:
                    return -1

            elif Keys.numpad_jump:
                if action in Actions.JUMPABLE_SET:
                    action += Actions.MOD_JUMP
                else:
                    return -1

            return action

import argparse

parser = argparse.ArgumentParser()

# common
parser.add_argument('-mw', '--mapwidth', type=int, help='Map Width, must be odd',
                    choices=range(9, 20, 2), metavar='[9-19 odd]',
                    default=9)

parser.add_argument('-mh', '--mapheight', type=int, help='Map Height, must be odd',
                    choices=range(13, 20, 2), metavar='[13-19 odd]',
                    default=15)

parser.add_argument('-d', '--difficulty', type=int,
                    help='Difficulty of the game, higher=more difficult',
                    choices=range(3), metavar='[0-2]',
                    default=1)

parser.add_argument('-s', "--seed", type=int,
                    help="The seed for random number generator",
                    default=None)

parser.add_argument('-fps', "--fps", type=int,
                    help="The rendering speed in frames per second",
                    default=15)

# for DQN
parser.add_argument('-r', '--render',
                    help="Render game state to screen",
                    action="store_true")

parser.add_argument('-e', "--episode", type=int, 
                    help="The number of episodes", 
                    default=10000)

parser.add_argument('-ms', "--maxstep", type=int,
                    help="The maximum number of steps in an episode",
                    default=500)

parser.add_argument('-f', "--file", type=str,
                    help="The file name of the DQN model",
                    default=None)

args = parser.parse_args()

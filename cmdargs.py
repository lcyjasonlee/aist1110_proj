import argparse

parser = argparse.ArgumentParser()

# common
parser.add_argument('-mw', '--mapwidth', type=int, help='Map Width', 
                    choices=range(5,20), metavar='[5-19]', 
                    default=9)

parser.add_argument('-mh', '--mapheight', type=int, help='Map Height', 
                    choices=range(5,20), metavar='[5-19]', 
                    default=15)

parser.add_argument('-d', '--difficulty', type=int, 
                    help='Difficulty of the Game, higher=more difficult', 
                    choices=range(3), metavar='[0-2]', 
                    default=1)

parser.add_argument('-s', "--seed", type=int, 
                    help="The seed for random number generator", 
                    default=None)

parser.add_argument('-fps', "--fps", type=int, 
                    help="The rendering speed in frames per second",
                    default=15)

# for DQN
parser.add_argument('-b', '--bot',
                    help="Bot mode, suppresses rendering",
                    action="store_true")

parser.add_argument('-e', "--episodes", type=int, 
                    help="The number of episodes.", 
                    default=1000)

parser.add_argument('-ms', "--max_steps", type=int, 
                    help="The maximum number of steps in an episode", 
                    default=500)

parser.add_argument('-f', "--file", type=str, 
                    help="The file name of the DQN Model",
                    default=None)

args = parser.parse_args()
import numpy as np
import gym
import the_floor_is_lava
from cmdargs import args
import pandas as pd


env = gym.make(
    "the_floor_is_lava-v1",
    map_width=args.mapwidth,
    map_height=args.mapheight,
    difficulty=args.difficulty, 
    render_mode="human" if args.render else None, 
    trunc=args.maxstep,
    seed=args.seed
)

scores = np.zeros(args.episode) # for plotting

for i in range(args.episode):
    done = False
    _, info = env.reset(seed=args.seed)
    while not done:
        action = env.action_space.sample()
        _, _, done, truncated, info = env.step(action)

        if truncated:
            done = True
    
    scores[i] = info["score"]
    
    if i % 100 == 0:
        print(f"episode {i} ended with a score of {info['score']}")


score_series = pd.Series(scores, name="score")
print(f"On average, random agent scores {score_series.mean()} per episode")

score_series.to_csv("data/random_score.csv", index=False)
import tensorflow as tf
import numpy as np
import gym
import the_floor_is_lava
from cmdargs import args
import pandas as pd
import re


if not args.file:
    raise ValueError("Path of DQN Model is Required")

# recover settings from file name
pattern = re.compile(r"the_floor_is_lava_w(\d*)_h(\d*)_d(\d*)")
match_obj = re.match(pattern, args.file)
map_width = int(match_obj.group(1))
map_height = int(match_obj.group(2))
difficulty = int(match_obj.group(3))

env = gym.make(
    "the_floor_is_lava-v1",
    map_width=map_width,
    map_height=map_height,
    difficulty=difficulty, 
    render_mode="human" if args.render else None, 
    trunc=args.maxstep,
    seed=args.seed,
    fps=args.fps if args.render else None
)

model = tf.keras.models.load_model(args.file)

scores = np.zeros(args.episode)

for i in range(args.episode):
    done = False
    obs, info = env.reset(seed=args.seed)
    while not done:
        prediction = model(obs[np.newaxis,:])[0]
        action = int(tf.math.argmax(prediction))
        new_obs, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True
    
    scores[i] = info["score"]
    
    print(f"episode {i}: score={info['score']}, step={info['step_count']}")


score_series = pd.Series(scores, name="score")
print(f"On average, DQN agent scores {score_series.mean()} per episode")

score_series.to_csv("data/dqn_score.csv", index=False)
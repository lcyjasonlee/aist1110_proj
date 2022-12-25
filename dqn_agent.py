import tensorflow as tf
import numpy as np
import gym
import the_floor_is_lava
from cmdargs import args
import pandas as pd

# model was trained with settings below
env = gym.make(
    "the_floor_is_lava-v1",
    map_width=9,
    map_height=15,
    difficulty=1, 
    render_mode="human" if args.render else None, 
    trunc=args.maxstep,
    seed=args.seed,
    fps=args.fps
)

model = tf.keras.models.load_model(args.file)

scores = np.zeros(args.episode)

for i in range(args.episode):
    done = False
    obs, info = env.reset(seed=args.seed)
    while not done:
        action = np.argmax(model.predict(obs[np.newaxis,:]))
        new_obs, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True
    
    scores[i] = info["score"]
    
    if i % 100 == 0:
        print(f"episode {i} ended with a score of {info['score']}")


score_series = pd.Series(scores, name="score")
print(f"On average, DQN agent scores {score_series.mean()} per episode")

score_series.to_csv("data/dqn_score.csv", index=False)
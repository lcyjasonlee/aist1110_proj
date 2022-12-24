from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress missing lib msg

# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import gym
import the_floor_is_lava
from cmdargs import args


env = gym.make(
    "the_floor_is_lava-v1",
    map_width=args.mapwidth,
    map_height=args.mapheight,
    difficulty=args.difficulty, 
    render_mode="human" if args.render else None, 
    trunc=args.maxstep
)

env.action_space.seed(args.seed)
observation, info = env.reset(seed=args.seed)

print(observation)
print(info)

action = env.action_space.sample()
observation, reward, done, truncated, info = env.step(action)

print(observation)
print(info)

# learning_rate = 0.001
# init = tf.keras.initializers.HeUniform(seed=3636798)
# model = keras.Sequential()
# model.add(keras.layers.Dense(4, input_shape=(3,), activation='relu', kernel_initializer=init))
# model.add(keras.layers.Dense(4, activation='relu', kernel_initializer=init))
# model.add(keras.layers.Dense(1, kernel_initializer=init))
# model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

# # print(model.summary())
# print(model.predict(np.array([[1,2,3], [4,5,6]])))
# many batches in 1 outer array
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress missing lib msg

import tensorflow as tf
from tensorflow import keras
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

# env.action_space.seed(args.seed)
# observation, info = env.reset(seed=args.seed)

# print(observation)
# print(info)

# action = env.action_space.sample()
# observation, reward, done, truncated, info = env.step(action)

# print(observation)
# print(info)

# stores & retrieve experiences
class ReplayMem:
    def __init__(self, size: int, shape: tuple) -> None:
        self.size = size      # no. of experience stored
        self.counter = 0  # for queue implementation
        
        self.obs_mem = np.zeros((size, *shape), dtype=np.float32)
        self.action = np.zeros((size,), dtype=np.int32)
        self.reward_mem = np.zeros((size,), dtype=np.int32)
        self.new_obs_mem = np.zeros((size, *shape), dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, reward: int, new_obs: np.ndarray) -> None:
        index = self.counter % self.size    # FIFO
        
        self.obs_mem[index] = obs
        self.action[index] = action
        self.reward_mem[index] = reward
        self.new_obs_mem[index] = new_obs
        
        self.counter += 1
        
    def sample(self, batch_size: int) -> tuple:
        if batch_size < self.counter:
            raise ValueError("Not Enough Experiences to Sample")

        rng = np.random.default_rng()
        sample_indices = rng.choice(
            min(self.counter, self.size),
            size=batch_size,
            replace=False
        )
        
        # return array of samples
        obs = self.obs_mem[sample_indices]
        action = self.action[sample_indices]
        reward = self.reward_mem[sample_indices]
        new_obs = self.new_obs_mem[sample_indices]
        
        return obs, action, reward, new_obs


class Network:
    # shape: tuple defining network structure,
    # input -> output from left to right
    # lr: learning rate for optimizer
    def __init__(self, shape: tuple, lr: float=0.001) -> None:
        
        init = tf.keras.initializers.HeUniform(seed=3636798)
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(4, input_shape=(3,), activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(4, activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(1, kernel_initializer=init))
        
        self.model.compile(loss=tf.keras.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])


# learn and apply strategy
class Agent:
    # eps: parameter on epsilon-greedy
    # in the format (max, min, decay rate)
    # discount: discount factor in bellman eq
    # lr: learning rate for updating q value
    def __init__(self, pnet: Network, tnet: Network, lr: float=0.7,
                 eps: tuple=(1, 0.01, 0.001), discount: float=0.999) -> None:
        self.policy_net = pnet
        self.target_net = tnet
        
        self.learning_rate = lr
        self.eps_max, self.eps_min, self.eps_decay = eps
        self.eps = self.eps_max     # decay over time
        self.discount_factor = discount
        
        
    
    def learn(self, obs: np.ndarray, action: int, reward: int, new_obs: np.ndarray) -> None:
        pass
    
    def act(self) -> int:
        pass
    

# learning_rate = 0.001
# init = tf.keras.initializers.HeUniform(seed=3636798)
# model = keras.Sequential()
# model.add(keras.layers.Dense(4, input_shape=(3,), activation='relu', kernel_initializer=init))
# model.add(keras.layers.Dense(4, activation='relu', kernel_initializer=init))
# model.add(keras.layers.Dense(1, kernel_initializer=init))
# model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
# print(model.predict(np.array([[1,2,3], [4,5,6]])))

# print(model.summary())
# many batches in 1 outer array
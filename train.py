from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress missing lib msg

import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import the_floor_is_lava
from cmdargs import args


if tf.test.gpu_device_name():
    print("Using GPU")


# stores & retrieve experiences
class ReplayMem:
    def __init__(self, obs_shape: tuple, size: int=1000) -> None:
        self.size = size      # no. of experience stored
        self.counter = 0  # for queue implementation
        
        self.obs_mem = np.zeros((size, *obs_shape), dtype=np.float32)
        self.action = np.zeros((size,), dtype=np.int32)
        self.reward_mem = np.zeros((size,), dtype=np.int32)
        self.new_obs_mem = np.zeros((size, *obs_shape), dtype=np.float32)
        self.done_mem = np.zeros((size,), dtype=np.bool_)

    def add(self, obs: np.ndarray, action: int, reward: int, new_obs: np.ndarray, done: bool) -> None:
        index = self.counter % self.size    # FIFO
        
        self.obs_mem[index] = obs
        self.action[index] = action
        self.reward_mem[index] = reward
        self.new_obs_mem[index] = new_obs
        self.done_mem[index] = done
        
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
        done = self.done_mem[sample_indices]
        
        return obs, action, reward, new_obs, done


# shape: tuple defining network structure,
# input -> output from left to right,
# assume len(shape) >= 3
# lr: learning rate for optimizer
def network(shape: tuple, lr: float=0.001) -> keras.Sequential:
    
    init = tf.keras.initializers.HeUniform(seed=3636798)
    model = keras.Sequential([
        keras.layers.Dense(shape[1], input_shape=shape[:1], 
                            activation="relu", kernel_initializer=init),
        *[keras.layers.Dense(i, activation="relu", 
                                kernel_initializer=init) for i in shape[1:-1]],
        keras.layers.Dense(shape[-1], kernel_initializer=init)
    ])
    
    # minDQN uses losses.Huber()
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    return model
    

# learn and apply strategy
class Agent:
    # eps: parameter on epsilon-greedy
    # in the format (max, min, decay rate)
    # discount: discount factor in bellman eq
    # lr: learning rate for updating q value
    # copy: no. of learnings before copying pnet to tnet
    # batch: no. of samples taken from rm when learning
    # a: no. of available actions
    def __init__(self, pnet: keras.Sequential, tnet: keras.Sequential, rm: ReplayMem, 
                 lr: float=0.7, discount: float=0.999, copy: int=10,
                 batch: int=100, eps: tuple=(1, 0.01, 0.001)) -> None:
        self.policy_net = pnet
        self.target_net = tnet
        self.replay_mem = rm
        self.action_count = pnet.output.shape[1]
        
        self.learning_rate = lr
        self.eps_max, self.eps_min, self.eps_decay = eps
        self.eps = self.eps_max     # decay over time
        self.discount_factor = discount
        self.step_count = 0
        
        self.batch_size = batch
        self.copy_threshold = copy
        self.learn_count = 0
        
        self.rng = np.random.default_rng(seed=args.seed)
    
    @property
    def exploration_rate(self) -> float:
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.eps_decay*self.step_count)
    
    # store experience & update network
    def learn(self, obs, action, reward, new_obs, done) -> None:
        self.replay_mem.add(obs, action, reward, new_obs, done)
        self.learn_count += 1
        
        try:
            obs_batch, action_batch, reward_batch, new_obs_batch, done_batch = \
                self.replay_mem.sample(self.batch_size)
        
            current_q = self.policy_net.predict(obs_batch)
            
            # max q value obtainable with next state
            max_q = np.max(self.target_net.predict(new_obs_batch), axis=1)
            
            # q value the optimal network should produce
            # if an action ends the game, expected q = reward obtained from that action
            optimal_q = reward_batch + self.discount_factor * (max_q * (1 - done_batch))
            
            # q values to be fitted to current observation
            rows = np.arange(self.batch_size)
            current_q[rows, action_batch] = \
                (1 - self.learning_rate) * current_q[rows, action_batch] \
                + self.learning_rate * optimal_q
            
            self.target_net.fit(obs_batch, current_q)
            
        except ValueError:
            return
        
        if self.learn_count % self.copy_threshold == 0:
            self.target_net.set_weights(self.policy_net.get_weights())

    def step(self, obs: np.ndarray) -> int:
        self.step_count += 1
        
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_count)
        else:
            return np.argmax(self.policy_net.predict(obs[np.newaxis,:])[0])
    


env = gym.make(
    "the_floor_is_lava-v1",
    map_width=args.mapwidth,
    map_height=args.mapheight,
    difficulty=args.difficulty, 
    render_mode="human" if args.render else None, 
    trunc=args.maxstep
)


oshape = env.observation_space.shape    # (21,)
qshape = env.action_space.n     # 22
network_shape = (oshape[0], oshape[0]*2, qshape*2, qshape)

pnet = network(network_shape)
tnet = network(network_shape)

agent = Agent(pnet, tnet, ReplayMem(oshape))


for _ in range(args.episode):
    done = False
    obs, info = env.reset(seed=args.seed)
    while not done:
        action = agent.step(obs)
        new_obs, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True
        
        agent.learn(obs, action, reward, new_obs, done)
        obs = new_obs
        
pnet.save("the_floor_is_lava_9x15")

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress missing lib msg

import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import the_floor_is_lava
from cmdargs import args
import pandas as pd
import gc


if tf.test.gpu_device_name():
    print("Using GPU")

tf.keras.utils.disable_interactive_logging()
tf.config.run_functions_eagerly(False)


class ReplayMem:
    """
    allows storage & retrieval of experiences
    """
    def __init__(self, obs_shape: tuple, size: int=50000) -> None:
        self.size = size      # no. of experience stored
        self.counter = 0  # for queue implementation
        
        self.obs_mem = np.zeros((size, *obs_shape), dtype=np.float32)
        self.action = np.zeros((size,), dtype=np.int32)
        self.reward_mem = np.zeros((size,), dtype=np.float32)
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
        if batch_size > self.counter:
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


def network(shape: tuple, lr: float=0.001) -> keras.Sequential:
    """
    create neural network
    
    parameters:
    shape: tuple defining network structure
           input -> output from left to right,
           assume len(shape) >= 3
           
    lr: learning rate for optimizer
    """

    model = keras.Sequential([
        keras.layers.Dense(shape[1], input_shape=shape[:1], 
                            activation="relu", kernel_initializer="he_uniform"),
        *[keras.layers.Dense(i, activation="relu", 
                                kernel_initializer="he_uniform") for i in shape[2:-1]],
        keras.layers.Dense(shape[-1], kernel_initializer="he_uniform")
    ])
    
    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    return model


class Agent:
    """
    learn and apply strategy
    """
    
    def __init__(self, pnet: keras.Sequential, tnet: keras.Sequential, rm: ReplayMem, 
                 lr: float=0.7, discount: float=0.999, copy: int=100,
                 batch: int=20, eps: tuple=(1, 0.01, 0.001)) -> None:
        """   
        parameters:
        eps: exploration rate & its decay parameter
             in the format (max, min, decay rate)
            
        discount: discount factor in bellman eq
        
        lr: learning rate for updating q value
            higher = recent values are more important
        
        copy: no. of learnings before copying pnet to tnet
        
        batch: no. of samples taken from rm when learning
        """
        
        self.policy_net = pnet
        self.target_net = tnet
        self.replay_mem = rm
        self.action_count = pnet.output.shape[1]
        
        self.learning_rate = lr
        self.eps_max, self.eps_min, self.eps_decay = eps
        self.discount_factor = discount
        self.step_count = 0
        
        self.batch_size = batch
        self.copy_threshold = copy
        self.learn_count = 0
        
        self.rng = np.random.default_rng(seed=args.seed)
        
        self.loss_history = []
    
    @property
    def exploration_rate(self) -> float:
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.eps_decay*self.step_count)
    
    # custom implementation for model.fit
    # workaround for memory leak, likely makes training slower
    @staticmethod
    def _fit(model, data_in, data_out) -> int:
        optimizer = model.optimizer
        loss_func = model.loss
        
        with tf.GradientTape() as tape:
            logits = model(data_in) 
            loss_value = loss_func(data_out, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        tf.keras.backend.clear_session() 
        gc.collect()
            
        return tf.reduce_mean(loss_value)
    
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
            
            loss = self._fit(self.policy_net, obs_batch, current_q)
            self.loss_history.append(int(loss))

        except ValueError:
            return
        
        if self.learn_count % self.copy_threshold == 0:
            self.target_net.set_weights(self.policy_net.get_weights())

    def step(self, obs: np.ndarray) -> int:
        self.step_count += 1
        
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_count)
        else:
            prediction = self.policy_net(obs[np.newaxis,:])[0]
            action = tf.math.argmax(prediction)
            return int(action)


env = gym.make(
    "the_floor_is_lava-v1",
    map_width=args.mapwidth,
    map_height=args.mapheight,
    difficulty=args.difficulty, 
    render_mode="human" if args.render else None,
    fps=args.fps if args.render else None,
    trunc=args.maxstep
)


oshape = env.observation_space.shape
qshape = env.action_space.n
network_shape = (oshape[0], qshape * 2, qshape)

if args.file:
    pnet = tf.keras.models.load_model(args.file)
    tnet = tf.keras.models.load_model(args.file)
else:
    pnet = network(network_shape)
    tnet = network(network_shape)



agent = Agent(pnet, tnet, ReplayMem(oshape))

scores = np.zeros(args.episode, dtype=np.int32) # for plotting
step_count = np.zeros(args.episode, dtype=np.int32)

for i in range(args.episode):
    done = False
    obs, info = env.reset(seed=args.seed)
    
    while not done:
        action = agent.step(obs)
        new_obs, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True
        
        agent.learn(obs, action, reward, new_obs, done)
        obs = new_obs
    
    scores[i] = info["score"]
    step_count[i] = info["step_count"]
    
    if i % 20 == 0:
        print(f"episode {i}: score={info['score']}, step={info['step_count']}, latest loss={agent.loss_history[-1:]}")


df = pd.DataFrame(data=
    {
        "score": scores,
        "step_count": step_count
    }
)
df.to_csv("data/train_stat.csv")

losses = pd.Series(agent.loss_history, name="loss")
losses.to_csv("data/train_loss.csv", index=False)

print(f"Average:  Score={df['score'].mean()}, Step={df['step_count'].mean()}")

pnet.save(f"the_floor_is_lava_w{args.mapwidth}_h{args.mapheight}_d{args.difficulty}")


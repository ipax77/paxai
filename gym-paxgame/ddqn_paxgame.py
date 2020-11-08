# https://rubikscode.net/2020/01/27/double-dqn-with-tensorflow-2-and-tf-agents-2/

import gym
import gym_paxgame
import os
import tensorflow as tf
from collections import deque

import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import History
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tempdir = "/data/ml"
policy_dir = os.path.join(tempdir, 'policy')
checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_step_counter = tf.Variable(0)
h5_file = '/data/ml/model_h5/ddqn_paxgame_1.h5'
DoLoadH5 = True

EPISODES = 2000

MAX_EPSILON = 1
MIN_EPSILON = 0.01

GAMMA = 0.95
LAMBDA = 0.0005
TAU = 0.08

BATCH_SIZE = 128
LAYERS = 800
REWARD_STD = 1.0

enviroment = gym.make("paxgame-v0")

NUM_STATES = 400
NUM_ACTIONS = enviroment.action_space.n
results = []

class ExpirienceReplay:
    def __init__(self, maxlen = 10000):
        self._buffer = deque(maxlen=maxlen)
    
    def store(self, state, action, reward, next_state, terminated):
        self._buffer.append((state, action, reward, next_state, terminated))
              
    def get_batch(self, batch_size):
        # f no_samples > len(self._samples):
        #    return random.sample(self._buffer, len(self._samples))
        # else:
        return random.sample(self._buffer, batch_size)
        
    def get_arrays_from_batch(self, batch):
        states = np.array([np.ravel(x[0]) for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        # next_states = np.array([(np.zeros(NUM_STATES) if x[3] is None else x[3]) 
        #                        for x in batch])
        next_states = np.array([(np.zeros(NUM_STATES) if x[3] is None else np.ravel(x[3])) 
                                 for x in batch])
        
        return states, actions, rewards, next_states
        
    @property
    def buffer_size(self):
        return len(self._buffer)

class DDQNAgent:
    def __init__(self, expirience_replay, state_size, actions_size, optimizer, DoLoadH5):
        
        # Initialize atributes
        self._state_size = state_size
        self._action_size = actions_size
        self._optimizer = optimizer
        self.DoLoadH5 = DoLoadH5
        self.expirience_replay = expirience_replay
        
        # Initialize discount and exploration rate
        self.epsilon = MAX_EPSILON
        
        # Build networks
        self.primary_network = self._build_network()
        self.primary_network.compile(loss='mse', optimizer=self._optimizer)

        self.target_network = self._build_network()   
   
    def _build_network(self):
        network = Sequential()
        network.add(Dense(LAYERS, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(LAYERS, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(self._action_size))
        
        return network
    
    def align_epsilon(self, step):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * step)
    
    def align_target_network(self):
        for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
            t.assign(t * (1 - TAU) + e * TAU)
    
    def act(self, state, mask):
        if np.random.rand() < self.epsilon:
            return np.random.choice([i for i in range(0, NUM_ACTIONS) if mask[i] != 1])
        else:
            q_values = np.ma.array(self.primary_network(state.reshape(1, -1)), mask=mask)
            return np.argmax(q_values)
    
    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.store(state, action, reward, next_state, terminated)
    
    def train(self, batch_size):
        if self.expirience_replay.buffer_size < BATCH_SIZE * 3:
            return 0
        
        batch = self.expirience_replay.get_batch(batch_size)
        states, actions, rewards, next_states = expirience_replay.get_arrays_from_batch(batch)
        
        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_values_state = self.primary_network(states).numpy()
        q_values_next_state = self.primary_network(next_states).numpy()
        
        # Copy the q_values_state into the target
        target = q_values_state
        updates = np.zeros(rewards.shape)
                
        valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(BATCH_SIZE)

        action = np.argmax(q_values_next_state, axis=1)
        q_next_state_target = self.target_network(next_states)
        updates[valid_indexes] = rewards[valid_indexes] + GAMMA * q_next_state_target.numpy()[batch_indexes[valid_indexes], action[valid_indexes]]
        
        target[batch_indexes, actions] = updates
        loss = self.primary_network.train_on_batch(states, target)

        # update target network parameters slowly from primary network
        self.align_target_network()
        
        if (self.DoLoadH5):
            self.primary_network.load_weights(h5_file)
            self.DoLoadH5 = False

        return loss

class AgentTrainer():
    def __init__(self, agent, enviroment):
        self.agent = agent
        self.enviroment = enviroment
        
    def _take_action(self, action):
        next_state, reward, terminated, _ = self.enviroment.step(action) 
        # next_state = next_state if not terminated else None
        # reward = np.random.normal(1.0, REWARD_STD)
        return next_state, reward, terminated
    
    def _print_epoch_values(self, episode, total_epoch_reward, average_loss):
        print("**********************************")
        print(f"Episode: {episode} - Reward: {total_epoch_reward} - Average Loss: {average_loss:.3f} - Epsilon: {agent.epsilon}")
        print(f"Rounds: {enviroment.state['round']} - Mins: {enviroment.state['minerals'][1]}|{enviroment.state['minerals'][-1]} - Hp: {enviroment.state['hp'][1]}|{enviroment.state['hp'][-1]}")
    
    def train(self, num_of_episodes = 1000):
        total_timesteps = 0  
        
        for episode in range(0, num_of_episodes):

            # Reset the enviroment
            state = self.enviroment.reset()

            # Initialize variables
            average_loss_per_episode = []
            average_loss = 0
            total_epoch_reward = 0
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
            step = 0
            terminated = False

            while not terminated:
                step += 1
                # Run Action
                action = agent.act(state, mask)

                # Take action    
                obs, reward, terminated = self._take_action(action)
                if (obs == None):
                    next_state = None
                else:
                    next_state = obs['state']
                    mask = obs['mask']
                agent.store(state, action, reward, next_state, terminated)
                
                loss = agent.train(BATCH_SIZE)
                average_loss += loss

                state = next_state
                # agent.align_epsilon(total_timesteps)
                agent.align_epsilon(episode)
                total_timesteps += 1

                if terminated:
                    preward = 0
                    if reward > 1.0:
                        preward = 1
                    elif reward < 1.0:
                        preward = -1
                    results.append(preward)
                    average_loss /= step
                    average_loss_per_episode.append(average_loss)
                    self._print_epoch_values(episode, total_epoch_reward, average_loss)
                
                # Real Reward is always 1 for Cart-Pole enviroment
                total_epoch_reward += reward

def moving(data, value=+1, size=100):
    binary_data = [x == value for x in data]
    # this is wasteful but easy to write...
    return [sum(binary_data[i-size:i])/size for i in range(size, len(data) + 1)]

def show(results, size=500, title='Moving average of game outcomes',
         first_label='First Player Wins', second_label='Second Player Wins', draw_label='Draw'):
    x_values = range(size, len(results) + 1)
    first = moving(results, value=+1, size=size)
    second = moving(results, value=-1, size=size)
    draw = moving(results, value=0, size=size)
    first, = plt.plot(x_values, first, color='red', label=first_label)
    second, = plt.plot(x_values, second, color='blue', label=second_label)
    draw, = plt.plot(x_values, draw, color='grey', label=draw_label)
    plt.xlim([0, len(results)])
    plt.ylim([0, 1])
    plt.title(title)
    plt.legend(handles=[first, second, draw], loc='best')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.ylabel(f'Rate over trailing window of {size} games')
    plt.xlabel('Game Number')
    plt.show()

optimizer = Adam()
expirience_replay = ExpirienceReplay(50000)
agent = DDQNAgent(expirience_replay, NUM_STATES, NUM_ACTIONS, optimizer, DoLoadH5)
agent_trainer = AgentTrainer(agent, enviroment)
agent_trainer.train(EPISODES)

agent.primary_network.save_weights('/data/ml/model_h5/ddqn_paxgame_1.h5')
show(results, size=int(EPISODES / 20))


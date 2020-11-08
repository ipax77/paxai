import gym
from gym import error, spaces, utils
from collections import deque,namedtuple
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Conv2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam
import requests
import json
import math
import datetime as dt
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker

class PaxGame:
    def __init__(self):
        self.env = gym.Env
        #self.env.action_space = spaces.Discrete(4800)
        #self.env.action_space = spaces.Discrete(40*120)
        self.env.action_space = spaces.Discrete(3*20*60 + 5)
        #self.env.reward_range: (-inf, inf)
        #self.env.observation_space = Box(4, 20, 50)
        
        self.env.state = np.zeros((20, 61))
        self.env.observation_space = spaces.Box(low=-2, high=2, shape=(20, 61))
        self.env.observation = np.zeros((20, 61))
    def reset(self):
        self.env.state = np.zeros((20, 61))
        self.env.observation = np.zeros((20, 61))

tf.keras.backend.set_floatx('float64')

p = PaxGame()
STORE_PATH = './TensorBoard'
MAXMINS = 3000
MAX_EPSILON = 1
MIN_EPSILON = 0.01
num_episodes = 10
EPSILON_MIN_ITER = 3
DELAY_TRAINING = 0
GAMMA = 0.95
BATCH_SIZE = 64
TAU = 0.08
RANDOM_REWARD_STD = 1.0
env = p.env
state_size = 20*61
num_actions = env.action_space.n

stepper = 10.0 ** 1

class DQModel(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQModel, self).__init__()
        self.dueling = dueling
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=tf.keras.initializers.he_normal())
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=tf.keras.initializers.he_normal())
        self.adv_dense = tf.keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=tf.keras.initializers.he_normal())
        self.adv_out = tf.keras.layers.Dense(num_actions,
                                          kernel_initializer=tf.keras.initializers.he_normal())
        if dueling:
            self.v_dense = tf.keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=tf.keras.initializers.he_normal())
            self.v_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal())
            self.lambda_layer = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = tf.keras.layers.Add()
    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(v)
            norm_adv = self.lambda_layer(adv)
            combined = self.combine([v, norm_adv])
            return combined
        return adv
    def get_config(self):
        pass

primary_network = DQModel(30, num_actions, True)
target_network = DQModel(30, num_actions, True)

#primary_network = DQModel(30, num_actions, False)
#target_network = DQModel(30, num_actions, False)

primary_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
# make target_network = primary_network
for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
    t.assign(e)

def update_network(primary_network, target_network):
    # update target network parameters slowly from primary network
    for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
        t.assign(t * (1 - TAU) + e * TAU)


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
    @property
    def num_samples(self):
        return len(self._samples)

memory = Memory(500000)

def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1), False
    else:
        return np.argmax(np.ma.array(primary_network(state.reshape(1, -1)), mask = player_objs[1].moves)), True



def train(primary_network, memory, target_network):
    batch = memory.sample(BATCH_SIZE)
    states = np.array([np.ravel(val[0]) for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else np.ravel(val[3])) for val in batch])
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards.astype(float)
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    # extract the best action from the next state
    prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
    # get all the q values for the next state
    q_from_target = target_network(next_states)
    # add the discounted estimated reward from the selected action (prim_action_tp1)
    updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    # update the q target to train towards
    target_q[batch_idxs, actions] = updates
    # run a training batch
    loss = primary_network.train_on_batch(states, target_q)
    return loss

def step(player, action, state, minerals):
    num = 0

    mystate = state.copy()
    myminerals = 0
    coord = player_objs[player].lastcoord

    if action == 3601:
        if mystate[0][60] == player:
            return mystate, myminerals, coord, False
        else:
            myminerals = 100 # Stimpack
            mystate[0][60] = player
            return mystate, myminerals, coord, True
    elif action == 3602:
        if mystate[1][60] == player:
            return mystate, myminerals, coord, False
        else:
            myminerals = 50 # CombatShield
            mystate[1][60] = player
            return mystate, myminerals, coord, True
    elif action == 3603:
        if mystate[2][60] == player:
            return mystate, myminerals, coord, False
        else:
            myminerals = 25 # ConcussiveShells
            mystate[2][60] = player
            return mystate, myminerals, coord, True
    elif action == 3604:
        if mystate[3][60] == player:
            return mystate, myminerals, coord, False
        else:
            myminerals = 100 # Attac Upgrade
            mystate[3][60] = player
            return mystate, myminerals, coord, True
    elif action == 3605:
        if mystate[4][60] == player:
            return mystate, myminerals, coord, False
        else:
            myminerals = 100 # Armor Upgrade
            mystate[4][60] = player
            return mystate, myminerals, coord, True

    for i in range(3):
        for j in range(20):
            for k in range(60):
                if (num == action):
                    x = i
                    y = j
                    z = k
                    unit = player + (x * 0.1)
                    if mystate[y][z] == unit:
                        return state, myminerals, coord, False
                    else:
                        if x == 0:
                            myminerals = 50  # Marine
                        elif x == 1:
                            myminerals = 95 # Marauder
                        elif x == 2:
                            myminerals = 65 # Reaper
                        
                    mystate[y][z] = unit
                    coord = np.array((y, z))
                num += 1
    return mystate, myminerals, coord, True

class Player():
    def __init__(self):
        self.minerals = 0
        self.moves = np.zeros(3*20*60 + 5)
        self.states = []
        self.actions = []
        self.rewardmod = []
        self.predicted = 0
        self.lastmove = 0
        self.lastcoord = np.array([])
        self.state = np.zeros((20, 61))
    def new_game(self):
        self.minerals = 0
        self.moves = np.zeros(3*20*60 + 5)
        self.states = []
        self.actions = []
        self.rewardmod = []
        self.predicted = 0
        self.lastmove = 0
        self.lastcoord = np.array([])
        self.state = np.zeros((20, 61))
    def move(self, primary_network, eps):
        move, predicted = choose_action(self.state, primary_network, eps)
        return move, predicted

class RandomPlayer(Player):
    def move(self, primary_network, eps):
        return random.randint(0, num_actions - 1), False

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

player_objs = {+1: Player(), -1: RandomPlayer()}
eps = MAX_EPSILON
#train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
steps = 0
results = []

for i in range(num_episodes):
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    fsminerals = MAXMINS
    p.reset()
    for player in [+1, -1]:
        player_objs[player].new_game()
    player = +1
    done = False
    resultcnt = 0
    headers = {'content-type': 'application/json'}
    if i == 5:
        primary_network.load_weights('model_h5/paxgame_gym_vs_dotnet_lineformation_1p5k_10k_upgradetest2.h5')

    while True:
        valid = False
        predicted = False
        action = -1
        minerals = 0
        coord = np.array([])
        while valid == False:
            action, predicted = player_objs[player].move(primary_network, eps)
            next_state, minerals, coord, valid = step(player, action, player_objs[player].state, player_objs[player].minerals)
            fsminerals -= minerals

        if action >= 0:
            player_objs[player].minerals += minerals
            player_objs[player].moves[action] = 1
            player_objs[player].actions.append(action)
            player_objs[player].states.append(next_state)
            if coord.any() and player_objs[player].lastcoord.any():
                dist = np.linalg.norm(player_objs[player].lastcoord - coord)
                #print (math.trunc(stepper * (int(dist) / 60)) / stepper)
                player_objs[player].rewardmod.append(math.trunc(stepper * (int(dist) / 60)) / stepper)
            else:
                player_objs[player].rewardmod.append(0)
            
            player_objs[player].lastcoord = coord

        if predicted == True:
            player_objs[player].predicted += 1

        reward = 0
        if player_objs[player].minerals >= MAXMINS:
            #print (player_objs[player].actions)
            #response = requests.post(url = "http://localhost:5000/get1dresult/" + str(MAXMINS), data = json.dumps(next_state.tolist()), headers = headers)  
            response = requests.post(url = "http://localhost:50586/get1dresult/" + str(MAXMINS), data = json.dumps(next_state.tolist()), headers = headers)  
            reward = float(response.text)
            preward = 0
            if reward >= 1:
                preward = 1
            elif reward <= -0.3:
                preward = -1
            results.append(preward)
            
            done = True
        #else:
        #    for s in range(int(MAXMINS / 500)):
        #        if (player_objs[player].minerals >= (s + 1) * 500 and resultcnt == s):
        #            resultcnt += 1
        #            response = requests.post(url = "http://localhost:5000/get1dresult/" + str((s + 1) * 500), data = json.dumps(next_state.tolist()), headers = headers)  
        #            reward = float(response.text)
        #            for r in range(len(player_objs[player].rewardmod)):
        #                player_objs[player].rewardmod[r] = reward
        


        tot_reward += reward

        if done:
            for j in range(len(player_objs[1].actions) - 1):
                memory.add_sample((player_objs[1].states[j], player_objs[1].actions[j], tot_reward - player_objs[player].rewardmod[j], player_objs[1].states[j+1]))
            memory.add_sample((next_state, player_objs[1].lastmove, tot_reward, None))
            if steps > DELAY_TRAINING:
                loss = train(primary_network, memory, target_network)
                update_network(primary_network, target_network)
            else:
                loss = -1
            avg_loss += loss

        
        #if player == 1 and action >= 0:
            # store in memory
            #memory.add_sample((player_objs[1].state, action, reward, next_state))
            #if steps > DELAY_TRAINING:
            #    loss = train(primary_network, memory, target_network)
            #    update_network(primary_network, target_network)
            #else:
            #    loss = -1
            #avg_loss += loss
            #avg_loss = loss

        if done:
            steps += 1
            if steps > DELAY_TRAINING:
                # linearly decay the eps value
                eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                    (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                    MIN_EPSILON
                #avg_loss /= cnt
                #print(f"Episode: {i}, Reward: {tot_reward}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                print(f"Episode: {i}, Reward: {tot_reward}, Predicted: {player_objs[1].predicted}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}, mins: {player_objs[player].minerals}")
                #with train_writer.as_default():
                #    tf.summary.scalar('reward', cnt, step=i)
                #    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print(f"Pre-training...Episode: {i}")
            break
        if action >= 0:
            player_objs[player].lastmove = action
            player_objs[player].state = next_state
        #player *= -1
        cnt += 1

primary_network.summary()
target_network.summary()

primary_network.save_weights('model_h5/paxgame_gym_vs_dotnet_lineformation_1p5k_10k_upgradetest3.h5')
primary_network.save('/data/ml/save/primary_network', save_format='tf')

collections.Counter(results)        
show(results, size=int(num_episodes / 20))
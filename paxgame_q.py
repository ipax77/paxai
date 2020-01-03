import random
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import requests
import json
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
checkpoint_path = "checkpoints/training1/pax_cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=0,
                                                 save_weights_only=True,
                                                 period=10
                                                 )

MAXMINS = 500
t1 = datetime.datetime.now()

def new_board(size):
    return np.zeros(shape=(4, 40, 120))

def available_moves(board):
    return np.argwhere(board == 0)

def step(move, board, player, minerals):

    myminerals = minerals
    mymove = move
    valid = False

    i = 0
    while valid == False:
        
        if i > 0:
            mymove = random.choice(available_moves(board))
            #if player == +1:
            #    print ('Random override prdiction')
        
        i += 1
        if player == +1 and (mymove[1] >= 20 or mymove[2] >= 60):
            continue

        if player == -1 and (mymove[1] < 20 or mymove[2] < 60):
            continue
        
        if board[mymove[0]][mymove[1]][mymove[2]] == player:
            continue

        else:
            x = mymove[0]
            y = mymove[1]
            z = mymove[2]
            if x == 1:
                myminerals -= 50  # Marine
            elif x == 2:
                myminerals -= 95  # Marauder
            elif x == 3:
                myminerals -= 65  # Reaper
            elif x == 0:
                if y == 0 and z == 1:
                    myminerals -= 100 # Stimpack
                elif y == 0 and  z == 2:
                    myminerals -= 50 # Combat Shield
                elif y == 0 and z == 3:
                    myminerals -= 25 # Concussive Shells
                elif y == 2 and z == 1:
                    myminerals -= 100 # Ground Attac lvl1
                elif y == 3 and z == 1:
                    myminerals -= 100 # Ground Armor lvl1
                else:
                    continue
        valid = True
        #if i == 1 and player == +1:
        #    print ('using prediction')
    return mymove, myminerals, valid

class Player():
    def __init__(self):
        self.minerals = MAXMINS
    def new_game(self):
        self.minerals = MAXMINS
    def reward(self, value):
        pass
    def train(self, move, board, value, q_values):
        pass

class RandomPlayer(Player):
    def move(self, board):
        return random.choice(available_moves(board)), 0, 0

def play(board, player_objs, count):
    print (count)
    for player in [+1, -1]:
        player_objs[player].new_game()
    player = +1
    game_end = None
    while game_end is None:
        
        valid = False
        myminerals = player_objs[player].minerals

        if player_objs[player].minerals > 0:
            while valid == False:
                move, value, q_values = player_objs[player].move(board)
                mymove, myminerals, valid = step(move, board, player, player_objs[player].minerals)
            
            player_objs[player].minerals = myminerals
            player_objs[player].train(mymove, board, value, q_values)
            board[tuple(mymove)] = player

        if player_objs[player].minerals <= 0 and player_objs[player * -1].minerals <= 0:
            headers = {'content-type': 'application/json'}
            response = requests.post(url = "http://localhost:5000/get4dresult/" + str(MAXMINS), data = json.dumps(board.tolist()), headers = headers)  
            #response = requests.post(url = "http://localhost:50586/get4dresult/" + str(MAXMINS), data = json.dumps(board.tolist()), headers = headers)  
            game_end = float(response.text)

        player *= -1  # switch players

    for player in [+1, -1]:
        reward_value = +1 if player == game_end else -1
        player_objs[player].reward(reward_value)
    return game_end
             
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


class BoringPlayer(Player):
    def move(self, board):
        return available_moves(board)[0]

class Agent(Player):
    def __init__(self, size, seed):
        self.size = size
        self.training = True
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(
            4*20*60,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
        self.model.compile(optimizer='sgd', loss='mean_squared_error')

    def predict_q(self, board):
        return self.model.predict(
            np.array([board.ravel()])).reshape(4, 20, 60)

    def fit_q(self, board, q_values):
        self.model.fit(
            #np.array([board.ravel()]), np.array([q_values.ravel()]), callbacks=[cp_callback], verbose=0)
            np.array([board.ravel()]), np.array([q_values.ravel()]), verbose=0)

    def new_game(self):
        self.last_move = None
        self.board_history = []
        self.q_history = []
        self.minerals = MAXMINS

    def move(self, board):
        # always ask the agent to play the same side
        q_values = self.predict_q(board)
        temp_q = q_values.copy()
        #temp_q[board != 0] = temp_q.min() - 1  # no illegal moves
        temp_q = np.argwhere(board == 0)
        #move = np.unravel_index(np.argmax(temp_q), board.shape)
        move = np.unravel_index(np.argmax(temp_q), (4, 20, 60))
        value = temp_q.max()
        return move, value, q_values

    def train(self, move, board, value, q_values):
        if self.training and self.last_move is not None:
            self.reward(value)
        self.board_history.append(board.copy())
        self.q_history.append(q_values)
        self.last_move = move

    def reward(self, reward_value):
        if not self.training:
            return
        new_q = self.q_history[-1].copy()
        #new_q[self.last_move] = reward_value
        new_q[self.last_move[0]][self.last_move[1]][self.last_move[2]] = reward_value
        self.fit_q(self.board_history[-1], new_q)



# 3x3, q-learning vs. random
random.seed(4)
agent = Agent(3, seed=4)

agent.model = tf.keras.models.load_model('model_h5/paxgame_q_500.h5')
agent.model.summary()

results = [play(new_board(3), {+1: agent, -1: RandomPlayer()}, i) for i in range(10000)]
collections.Counter(results)

agent.model.save('model_h5/paxgame_q_500.h5')
t2 = datetime.datetime.now()
print (t2 - t1)

show(results, size=500)
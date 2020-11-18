from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import requests
import json
import random
import time

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

BUNITS = 3
BSIZEX = 10
BSIZEY = 18
BUPGRADES = 5
BPOS = BUNITS * BSIZEX * BSIZEY
BSIZE = BSIZEX * BSIZEY
num_actions = BUNITS*BSIZEX*BSIZEY + BUPGRADES
STEPMINERALS = 500
MAXHP = 2000    

headers = {'content-type': 'application/json'}
seperator = ','
with open('/data/pgconfig.json', 'r') as f:
    config = json.load(f)
# RESTurl = config['pgConfig']['NetResturl']
# RESTurl = "http://localhost:51031/"
RESTurl = "http://localhost:5000/"
withMask = False

class PaxGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=num_actions, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(20,20), dtype=np.uint8, minimum=0, maximum=20, name='observation')
        self._state_spec = array_spec.BoundedArraySpec(shape=(20,20), dtype=np.uint8, minimum=0, maximum=20, name='state')
        self._mask_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=num_actions, name='mask')
        # self._observation_spec = { 'state': self._state_spec, 'mask': self._mask_spec }
        self._state = np.zeros((20, 18 + 2), dtype=np.uint8)
        self.mask = {}
        self.mask[1] = np.ones(num_actions + 1, dtype=bool)
        self.mask[-1] = np.ones(num_actions + 1, dtype=bool)
        self.hp = {}
        self.hp[1] = 0
        self.hp[-1] = 0
        self.minerals = {}
        self.minerals[1] = 0
        self.minerals[-1] = 0
        self.actions = {}
        self.actions[1] = []
        self.actions[-1] = []
        self.player = 1
        self.round = 1
        self.build = None
        self.monobuild = False
        self._episode_ended = False

        self.unitmask = {}
        for i in range(BUNITS):
            self.unitmask[i] = np.zeros(num_actions, dtype=np.int8)

        for i in range(BPOS):
            bunit = int(i / BSIZE)
            for j in range(BUNITS):
                if bunit != j:
                    self.unitmask[j][i] = 1

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_masked_legal_actions(self, player):
        return self.mask[player]

    def _reset(self):
        self._state = np.zeros((20, 18 + 2), dtype=np.uint8)
        self.mask[1] = np.ones(num_actions + 1, dtype=bool)
        self.mask[-1] = np.ones(num_actions + 1, dtype=bool)
        self.hp[1] = MAXHP
        self.hp[-1] = MAXHP
        self.minerals[1] = 0
        self.minerals[-1] = 0
        self.actions[1] = []
        self.actions[-1] = []
        self.player = 1
        self.round = 1
        self.build = None
        self.monobuild = False
        self._episode_ended = False
        self._episode_ended = False
        if withMask:
            obs = {'state': self._state, 'mask': self.mask[1]}
            return obs
        else:
            return ts.restart(self._state)

    def _step(self, action, player=1, isSinglePlayer=True):

        if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
            return self.reset()

        self.player = player
        move = action.item(0)
        reward = 0.0
        roundmins = self.round * STEPMINERALS
        isvalid = self.state_generator(move)
        
        if isvalid:
            self.actions[player].append(move)
        else:
            self._episode_ended = True
            reward = -10
            return ts.termination(self._state, reward)

        if isSinglePlayer and self.minerals[player] >= roundmins:

            if isSinglePlayer:
                self.GenRandomBuild(self.build, self.monobuild, roundmins)

            restsuccess = False
            request = RESTurl + "paxgame/result/" + seperator.join(str(x) for x in self.actions[1]) + "/" + seperator.join(str(x) for x in self.actions[-1])
            try:
                response = requests.get(url = request)
                restsuccess = True
            except:
                print("Socket error :((")
                time.sleep(1)
                i = 0
                while i < 10:
                    try:
                        response = requests.get(url = request)
                        restsuccess = True
                    except:
                        i += 1
                        time.sleep(2)
            finally:
                if restsuccess == False:
                    print("Socket error retry failed.")

            responses = response.text.split(seperator)
            if self.player == 1:
                reward += float(responses[0])
            elif self.player == -1:
                reward += float(responses[1])
            
            self.hp[1] += int(responses[2])
            self.hp[-1] += int(responses[3])
            if self.round > 15:
                if self.hp[1] > self.hp[-1]:
                    self.hp[1] = MAXHP
                else:
                    self.hp[-1] = MAXHP
            if self.hp[1] >= MAXHP or self.hp[-1] >= MAXHP:
                if self.hp[1] >= MAXHP:
                    if self.player == 1:
                        reward -= 1
                    else:
                        reward += 1
                else:
                    if self.player == 1:
                        reward += 1
                    else:
                        reward -= 1
                self._episode_ended = True

        if not isSinglePlayer:
            if (self.minerals[1] >= roundmins and self.minerals[-1] < roundmins):
                self.player = -1
            elif (self.minerals[-1] >= roundmins and self.minerals[1] < roundmins):
                self.player = 1
            else:    
                self.player = -1  

        if reward == 0.0:
            reward = 0.1

        if (self.minerals[1] >= roundmins and self.minerals[-1] >= roundmins):
            self.round += 1            

        if self._episode_ended:
            if withMask:
                obs = {'state': ts.termination(self._state, reward), 'mask': self.mask[self.player]}
                return obs
            else:
                return ts.termination(self._state, reward)
        else:
            if withMask:
                obs = {'state': ts.transition(self._state, reward=reward, discount=0.9), 'mask': self.mask[self.player]}
                return obs
            else:
                return ts.transition(self._state, reward=reward, discount=0.9)

    def state_generator(self, move):
        if move >= num_actions:
            return False
        minerals = 0
        valid = True
        if move >= BPOS:
            ability = move - BPOS
            if ability == 0:
                minerals = 100 # Stimpack
            elif ability == 1:
                minerals = 50 # CombatShield
            elif ability == 2:
                minerals = 25 # ConcussiveShells
            elif ability == 3:
                minerals = 100 # attac
            elif ability == 4:
                minerals = 100 # armor
            if self.player == 1:
                if self._state[18][ability] == 0:
                    self._state[18][ability] = 4
                else:
                    valid = False
            else:
                if self._state[18][(ability - BSIZEY) * -1] == 0:
                    self._state[18][(ability - BSIZEY) * -1] = 14
                else:
                    valid = False
        else:
            unitid = int(move / BSIZE)
            mymod = int(move - (unitid * BSIZE))
            x = int(mymod / int(BSIZEY))
            y = int(mymod - (int(mymod / BSIZEY) * BSIZEY))
            if unitid == 0:
                minerals = 50 # Marine
            elif unitid == 1:
                minerals = 95 # Marauder
            elif unitid == 2:
                minerals = 65
            if self.player == 1:
                if self._state[y][x] == 0:
                    self._state[y][x] = int((unitid + 1))
                else:
                    valid = False
            elif self.player == -1:
                (mx, my) = self.State_mirrorImage(x, y)
                if self._state[int(my)][int(mx)] == 0:
                    self._state[int(my)][int(mx)] = int((unitid + 1) + 10)
                else:
                    valid = False
        if valid == False:
            minerals = 0
        else:
            p = self.player
            moves = self.mask[p]
            moves[move] = 0
            aent = move + BSIZE
            while aent < num_actions:
                moves[aent] = 0
                aent += BSIZE
            aent = move - BSIZE
            while aent >= 0:
                moves[aent] = 0
                aent -= BSIZE
        self.minerals[self.player] += minerals
        return valid

    def render(self, mode='human', close=False):
        if close:
            return
        print("Round: " + str(self.round) + " OnMove: " + str(self.player) + " Mins1: " + str(self.minerals[1]) + " Mins2: " + str(self.minerals[-1]))
        # print(self.state['board'])
        for i in range (BSIZEY + 1):
           print("|", end="")
           for j in range (BSIZEX * 2):
               #print(str(self.state['board'][i, j]) + "|", end="")
               print(str(self._state[i][j]) + "|", end="")
           print()

    def GenRandomBuild(self, build, mono, minerals):
        self.player = -1
        line = -1
        linemoves = []
        move = -1
        if not build:
            build = str(random.choice(['random', 'line', 'dist']))
            mono = bool(random.getrandbits(1))

        if build == 'random' and mono == False:
            possiblemoves = [i for i in range(0, num_actions) if self.mask[-1][i] != 0]
        else:
            monounit = -1
            for i in range(len(self.actions[-1])):
                if (self.actions[-1][i] <= BPOS):
                    monounit = int(self.actions[-1][i] / BSIZE)
                    if monounit > 2:
                        monounit = 2
                    mymod = int(i - (monounit * BSIZE))
                    line = int(mymod / int(BSIZEY))
                    move = self.actions[-1][i]
                    break
            if monounit == -1:
                monounit = int(np.random.randint(low=0, high=BUNITS - 1, size=1))
                line = int(np.random.randint(low=0, high=BSIZEX - 1, size=1))
                move = int(np.random.randint(low=0, high=BPOS - 1, size=1))
            possiblemoves = [i for i in range(0, num_actions) if self.mask[-1][i] != 0 and self.unitmask[monounit][i] == 0]

        while self.minerals[-1] < minerals:
            if build == 'random':
                rmove = random.choice([i for i in possiblemoves if self.mask[-1][i] != 0])
            elif build == 'line':
                linemoves = [i for i in possiblemoves if self.mask[-1][i] != 0 and self.GetLine(i, line)]
                while not linemoves:
                    line += 1
                    if line > BSIZEX -1:
                        line = 0
                    linemoves = [i for i in possiblemoves if self.mask[-1][i] != 0 and self.GetLine(i, line)]
                rmove = random.choice(linemoves)
            elif build == 'dist':
                distmoves = [i for i in self.GetDistMoves(move, possiblemoves) if self.mask[-1][i] != 0]
                rmove = random.choice(distmoves)

            isvalid = self.state_generator(rmove)
            if isvalid:
                self.actions[-1].append(rmove)

        self.player = 1

    def GetLine(self, move, line):
        if move > BPOS:
            return True
        unit = int(move / BSIZE)
        mymod = int(move - (unit * BSIZE))
        return int(mymod / int(BSIZEY)) == line

    def GetDistMoves(self, move, moves):
        point = self.GetXY(move)
        points = np.asarray([self.GetXY(i) for i in moves])
        dists = (points - point)**2
        dists = np.sum(dists, axis=1)
        dists = np.sqrt(dists)
        dists, moves = zip(*sorted(zip(dists, moves)))
        return moves[:20]

    def GetXY(self, move):
        unit = int(move / BSIZE)
        mymod = int(move - (unit * BSIZE))
        x = int(mymod / int(BSIZEY))
        y = int(mymod - (int(mymod / BSIZEY) * BSIZEY))
        return [x, y]

    def State_mirrorImage(self, x1, y1):
        a = 1
        b = 0
        c = (BSIZEX * 2) / -2
        temp = -2 * (a * x1 + b * y1 + c) / (a * a + b * b)
        x = temp * a + x1
        y = temp * b + y1
        return (x - 1, y)
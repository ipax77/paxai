from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import requests
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
RESTurl = "http://localhost:5000/"

class PaxGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action')
        self._state_spec = array_spec.ArraySpec(shape=(1, 20, 20), dtype=np.int32, name='state')
        self._mask_spec = array_spec.ArraySpec(shape=(BUNITS*BSIZEX*BSIZEY + BUPGRADES, ), dtype=np.int32, name='mask')
        self._mask = {}
        self._mask[1] = np.ones(BUNITS*BSIZEX*BSIZEY + BUPGRADES)
        self._mask[-1] = np.ones(BUNITS*BSIZEX*BSIZEY + BUPGRADES)
        self._state = np.zeros((20, 18 + 2), dtype=np.int32)
        self._observation_spec = self._state_spec
        self.on_move = 1
        self.hp = {}
        self.hp[1] = 0
        self.hp[-1] = 0
        self.minerals = {}
        self.minerals[1] = 0
        self.minerals[-1] = 0
        self.round = 1
        self.actions = {}
        self.actions[1] = []
        self.actions[-1] = []
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((20, 18 + 2), dtype=np.int32)
        self._mask[1] = np.ones(BUNITS*BSIZEX*BSIZEY + BUPGRADES)
        self._mask[-1] = np.ones(BUNITS*BSIZEX*BSIZEY + BUPGRADES)
        self.on_move = 1
        self.hp = {}
        self.hp[1] = 0
        self.hp[-1] = 0
        self.minerals = {}
        self.minerals[1] = 0
        self.minerals[-1] = 0
        self.round = 1
        self.actions = {}
        self.actions[1] = []
        self.actions[-1] = []
        self._episode_ended = False
        obs = { 'state': self._state, 'mask': self._mask[1] }
        # return obs
        return ts.restart(np.array([self._state], dtype=np.int32))
        # return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        isSinglePlayer = False
        p = 1
        # if (type(action) is int):
        if True:
            move = action
            self.on_move = p
            isSinglePlayer = True
        else:
            p, move = action
            self.on_move = p
        
        reward = 0
        roundmins = self.round * STEPMINERALS
        isvalid = self.state_generator(move)
        if isvalid:
            self.actions[p].append(str(move))
        else:
            # return self._state, -10, True, {}
            obs = self._state
            mask = self._mask[1]
            obs = {'state': obs, 'mask': mask}
            self._episode_ended = True
            return ts.termination(obs, -10)

        if ((isSinglePlayer and self.minerals[p] >= roundmins) or (self.minerals[p] >= roundmins and self.minerals[-p] >= roundmins)):
            if isSinglePlayer:
                while self.minerals[-1] < roundmins:
                    rmove = random.choice([i for i in range(0, num_actions - 1) if i not in self.actions[-1]])
                    self.on_move = -1
                    isvalid = self.state_generator(rmove)
                    if isvalid:
                        self.actions[-1].append(str(rmove))
                self.on_move = 1
            restsuccess = False
            request = RESTurl + "paxgame/result/" + seperator.join(self.actions[1]) + "/" + seperator.join(self.actions[-1])
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
            if p == 1:
                reward += float(responses[0])
            elif p == -1:
                reward += float(responses[1])
            
            self.hp[1] += int(responses[2])
            self.hp[-1] += int(responses[3])
            if self.hp[1] >= MAXHP or self.hp[-1] >= MAXHP:
                if self.hp[1] >= MAXHP:
                    if self.on_move == 1:
                        reward -= 1
                    else:
                        reward += 1
                else:
                    if self.on_move == 1:
                        reward += 1
                    else:
                        reward -= 1
                self._episode_ended = True

        if (self.minerals[1] >= roundmins and self.minerals[-1] < roundmins):
            self.on_move = -1
        elif (self.minerals[-1] >= roundmins and self.minerals[1] < roundmins):
            self.on_move = 1
        else:    
            self.on_move = -1       

        if (self.minerals[1] >= roundmins and self.minerals[-1] >= roundmins):
            self.round += 1
        #DEBUG
        # if (done == True):
        #    print("Reward: ", reward1, " hp1: ", hp1, " hp2: ", hp2)
        #    print("Mins1: ", mins1, " Mins2: ", mins2, " Round: ", self.round)
        #    self.render()

        if (reward == 0):
            reward = 0.1
        # return self._state, reward, done, {}
        obs = self._state
        mask = self._mask[1]
        obs = {'state': obs, 'mask': mask}
        if (self._episode_ended):
            return ts.termination(np.array([self._state], dtype=np.int32), reward=reward)
        else:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=reward, discount=0.1)

    def state_generator(self, move):
        minerals = 0
        valid = True
        if move > BPOS:
            ability = move - BPOS
            if ability == 1:
                minerals = 100 # Stimpack
            elif ability == 2:
                minerals = 50 # CombatShield
            elif ability == 3:
                minerals = 25 # ConcussiveShells
            elif ability == 4:
                minerals = 100 # attac
            elif ability == 5:
                minerals = 100 # armor
            if self.on_move == 1:
                if self._state[-2, ability] == 0:
                    self._state[-2, ability] = int(1)
                else:
                    valid = False
            else:
                if self._state[-2, (ability - BSIZEY) * -1] == 0:
                    self._state[-2, (ability - BSIZEY) * -1] = int(2)
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
            if self.on_move == 1:
                if self._state[y][x] == 0:
                    self._state[y][x] = int((unitid + 1) + 10)
                else:
                    valid = False
            elif self.on_move == -1:
                (mx, my) = self.State_mirrorImage(x, y)
                if self._state[int(my)][int(mx)] == 0:
                    self._state[int(my)][int(mx)] = int((unitid + 1) + 20)
                else:
                    valid = False
        if valid == False:
            minerals = 0
        else:
            p = self.on_move
            moves = self._mask[p]
            moves[move] = 0
            if move < BPOS:
                taction = move + BSIZEX * BSIZEY
                while taction < BPOS:
                    moves[taction] = 0
                    taction += BSIZEX * BSIZEY
                taction = move - BSIZEX * BSIZEY
                while taction >= 0:
                    moves[taction] = 0
                    taction -= BSIZEX * BSIZEY
        self.minerals[self.on_move] += minerals
        return valid

    def State_mirrorImage(self, x1, y1):
        a = 1
        b = 0
        c = (BSIZEX * 2) / -2
        temp = -2 * (a * x1 + b * y1 + c) / (a * a + b * b)
        x = temp * a + x1
        y = temp * b + y1
        return (x - 1, y)
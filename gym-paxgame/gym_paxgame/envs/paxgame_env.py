import gym
from gym import spaces
import numpy as np
import requests
import json
import time
import random
from tf_agents.specs import array_spec

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
StateWithMask = True

class PaxGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    symbols = ['O', ' ', 'X'];

    def __init__(self):
        self.action_space = spaces.Discrete(num_actions)
        # self.build_area_player1 = spaces.Box(low=0, high=2000, shape=(10, 20), dtype=np.int32)
        # self.build_area_player2 = spaces.Box(low=0, high=2000, shape=(10, 20), dtype=np.int32)
        # self.observation_space = spaces.Tuple([self.build_area_player1, self.build_area_player2])
        self.observation_space = spaces.Box(low=0, high=2000, shape=(20, 20), dtype=np.int32)

        self.unitmask = {}
        for i in range(BUNITS):
            self.unitmask[i] = np.zeros(num_actions, dtype=np.int8)

        for i in range(BPOS):
            bunit = int(i / BSIZE)
            for j in range(BUNITS):
                if bunit != j:
                    self.unitmask[j][i] = 1

    def get_osbervation_spec(self):
        """:returns An 'arrayspec' or a nested dict, list or tuple"""
        state_spec = array_spec.ArraySpec(shape=(20, 20), dtype=np.int32, name='state')
        mask_spec = array_spec.ArraySpec(shape=(BUNITS*BSIZEX*BSIZEY + BUPGRADES, ), dtype=np.int32, name='mask')
        obs = {state_spec, mask_spec}
        return obs

    def step(self, action):
        done = False
        isSinglePlayer = False
        p = 1
        # if (type(action) is int):
        if (True):
            move = action
            self.state['on_move'] = p
            isSinglePlayer = True
        else:
            p, move = action
            self.state['on_move'] = p
        
        reward = 0
        roundmins = self.state['round'] * STEPMINERALS
        isvalid = self.state_generator(move)
        if isvalid:
            self.state['actions'][p].append(move)
        else:
            print("invalid action :( - " + str(move))
            # return self.state['board'], -10, True, {}
            # obs = self.state['board']
            # mask = self.state['moves'][1]
            # obs = {'state': obs, 'mask': mask}
            # return obs, -10, True, {}
            # reward = -0.5

        if ((isSinglePlayer and self.state['minerals'][p] >= roundmins) or (self.state['minerals'][p] >= roundmins and self.state['minerals'][-p] >= roundmins)):
            if isSinglePlayer:
                self.GenRandomBuild(self.build, self.mono, roundmins)
                # print(np.sort(self.state['actions'][-1]))
            restsuccess = False
            request = RESTurl + "paxgame/result/" + seperator.join(str(x) for x in self.state['actions'][1]) + "/" + seperator.join(str(x) for x in self.state['actions'][-1])
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
            
            self.state['hp'][1] += int(responses[2])
            self.state['hp'][-1] += int(responses[3])
            if self.state['round'] > 15:
                if self.state['hp'][1] > self.state['hp'][-1]:
                    self.state['hp'][1] = MAXHP
                else:
                    self.state['hp'][-1] = MAXHP
            if self.state['hp'][1] >= MAXHP or self.state['hp'][-1] >= MAXHP:
                if self.state['hp'][1] >= MAXHP:
                    if self.state['on_move'] == 1:
                        reward -= 1
                    else:
                        reward += 1
                else:
                    if self.state['on_move'] == 1:
                        reward += 1
                    else:
                        reward -= 1
                done = True

        if (self.state['minerals'][1] >= roundmins and self.state['minerals'][-1] < roundmins):
            self.state['on_move'] = -1
        elif (self.state['minerals'][-1] >= roundmins and self.state['minerals'][1] < roundmins):
            self.state['on_move'] = 1
        else:    
            self.state['on_move'] = -1       

        if (self.state['minerals'][1] >= roundmins and self.state['minerals'][-1] >= roundmins):
            self.state['round'] += 1
        #DEBUG
        # if (done == True):
        #    print("Reward: ", reward1, " hp1: ", hp1, " hp2: ", hp2)
        #    print("Mins1: ", mins1, " Mins2: ", mins2, " Round: ", self.state['round'])
        #    self.render()

        if (reward == 0):
            reward = 0.1
        
        if (StateWithMask): 
            obs = self.state['board']
            mask = self.state['moves'][1]
            obs = {'state': obs, 'mask': mask}
            return obs, reward, done, {}
        else:
            return self.state['board'], reward, done, {}
    def reset(self):
        self.state = {}
        self.state['board'] = np.zeros((20, 18 + 2), dtype=np.int32)
        self.state['on_move'] = 1
        self.state['moves'] = {}
        self.state['moves'][1] = np.zeros(BUNITS*BSIZEX*BSIZEY + BUPGRADES, dtype=np.int8)
        self.state['moves'][-1] = np.zeros(BUNITS*BSIZEX*BSIZEY + BUPGRADES, dtype=np.int8)
        self.state['hp'] = {}
        self.state['hp'][1] = 1
        self.state['hp'][-1] = 1
        self.state['minerals'] = {}
        self.state['minerals'][1] = 0
        self.state['minerals'][-1] = 0
        self.state['round'] = 1
        self.state['actions'] = {}
        self.state['actions'][1] = []
        self.state['actions'][-1] = []
        self.build = None
        self.mono = False
        # return np.array_split(self.state['board'], 2, axis=1)
        return self.state['board']
    def render(self, mode='human', close=False):
        if close:
            return
        print("Round: " + str(self.state['round']) + " OnMove: " + str(self.state['on_move']) + " Mins1: " + str(self.state['minerals'][1]) + " Mins2: " + str(self.state['minerals'][-1]))
        # print(self.state['board'])
        for i in range (BSIZEY + 1):
           print("|", end="")
           for j in range (BSIZEX * 2):
               #print(str(self.state['board'][i, j]) + "|", end="")
               print(str(self.state['board'][i, j]) + "|", end="")
           print()

    def move_generator(self, om):
        moves = [i for i in range(0, num_actions - 1) if self.state['moves'][om][i] == 0]
        return moves
    
    def get_action_mask(self):
        return self.state['moves'][1]

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
            if self.state['on_move'] == 1:
                if self.state['board'][-2, ability] == 0:
                    self.state['board'][-2, ability] = int(1)
                else:
                    valid = False
            else:
                if self.state['board'][-2, (ability - BSIZEY) * -1] == 0:
                    self.state['board'][-2, (ability - BSIZEY) * -1] = int(2)
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
            if self.state['on_move'] == 1:
                if self.state['board'][y][x] == 0:
                    self.state['board'][y][x] = int((unitid + 1) + 10)
                else:
                    valid = False
            elif self.state['on_move'] == -1:
                (mx, my) = self.State_mirrorImage(x, y)
                if self.state['board'][int(my)][int(mx)] == 0:
                    self.state['board'][int(my)][int(mx)] = int((unitid + 1) + 20)
                else:
                    valid = False
        if valid == False:
            minerals = 0
        else:
            p = self.state['on_move']
            moves = self.state['moves'][p]
            moves[move] = 1
            aent = move + BSIZE
            while aent < num_actions:
                moves[aent] = 1
                aent += BSIZE
            aent = move - BSIZE
            while aent >= 0:
                moves[aent] = 1
                aent -= BSIZE
        self.state['minerals'][self.state['on_move']] += minerals
        return valid
    
    def GenRandomBuild(self, build, mono, minerals):
        self.state['on_move'] = -1
        line = -1
        linemoves = []
        move = -1
        if not build:
            build = str(random.choice(['random', 'line', 'dist']))
            mono = bool(random.getrandbits(1))

        if build == 'random' and mono == False:
            possiblemoves = [i for i in range(0, num_actions) if self.state['moves'][-1][i] != 1]
        else:
            monounit = -1
            for i in range(len(self.state['actions'][-1])):
                if (self.state['actions'][-1][i] <= BPOS):
                    monounit = int(self.state['actions'][-1][i] / BSIZE)
                    mymod = int(i - (monounit * BSIZE))
                    line = int(mymod / int(BSIZEY))
                    move = self.state['actions'][-1][i]
                    break
            if monounit == -1:
                monounit = int(np.random.randint(low=0, high=BUNITS - 1, size=1))
                line = int(np.random.randint(low=0, high=BSIZEX - 1, size=1))
                move = int(np.random.randint(low=0, high=BPOS - 1, size=1))
            possiblemoves = [i for i in range(0, num_actions) if self.state['moves'][-1][i] != 1 and self.unitmask[monounit][i] == 0]

        while self.state['minerals'][-1] < minerals:
            if build == 'random':
                rmove = random.choice([i for i in possiblemoves if self.state['moves'][-1][i] != 1])
            elif build == 'line':
                linemoves = [i for i in possiblemoves if self.state['moves'][-1][i] != 1 and self.GetLine(i, line)]
                while not linemoves:
                    line += 1
                    if line > BSIZEX -1:
                        line = 0
                    linemoves = [i for i in possiblemoves if self.state['moves'][-1][i] != 1 and self.GetLine(i, line)]
                rmove = random.choice(linemoves)
            elif build == 'dist':
                distmoves = [i for i in self.GetDistMoves(move, possiblemoves) if self.state['moves'][-1][i] != 1]
                rmove = random.choice(distmoves)

            isvalid = self.state_generator(rmove)
            if isvalid:
                self.state['actions'][-1].append(rmove)

        self.state['on_move'] = 1
    
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





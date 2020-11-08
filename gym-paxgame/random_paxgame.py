import math
import random
import gym_paxgame

import gym


def random_plus_middle_move(moves):
    m = random_move(moves)
    return m


def random_move(moves):
    m = random.choice(moves)
    return m


env = gym.make('paxgame-v0')

p1 = 0.48
p2 = 0.55
alpha = 0.01
beta = 0.01
# theta = math.log((p1*(1-p0)) / (p0*(1-p1)));

h1 = math.log((1 - alpha) / beta) / (math.log(p2 / p1) + math.log((1 - p1) / (1 - p2)))
h2 = math.log((1 - beta) / alpha) / (math.log(p2 / p1) + math.log((1 - p1) / (1 - p2)))
ss = math.log((1 - p1) / (1 - p2)) / (math.log(p2 / p1) + math.log((1 - p1) / (1 - p2)))
print("ss:", ss)
print("h1:", h1)
print("h2:", h2)

num_episodes = 100
num_steps_per_episode = 10000

collected_rewards = []
oom = 1
win1 = 0
win2 = 0
for i in range(num_episodes):
    s = env.reset()
    # print (s)
    # print ("starting new episode")
    # env.render()
    # print ("started")
    total_reward = 0
    done = False
    om = oom;
    # run one episode
    # print("starting player: ", om);

    for j in range(num_steps_per_episode):
        moves = env.move_generator(om)
        # print ("moves: ", moves)
        if (not moves):
            # print ("out of moves")
            break
        if (len(moves) == 1):
            # only a single possible move
            m = moves[0]
        else:
            if (om == 1):
                m = random_plus_middle_move(moves)
                 #m = random_move(moves, om)
            else:
                m = random_move(moves)
        # print ("m: ", m)
        # s1, reward1, done, _ = env.step([i, om, m])
        # s1, reward1, done, _ = env.step([om, m])
        s1, reward1, done, _ = env.step(m)
        # s1, reward1, done, _ = env.step([-1, 385])
        # print(m)
        # env.render()
        # om = -om
        om = env.state['on_move']
        # env.render()
        total_reward += reward1
        s = s1
        if done:
            # print ("game over: ", reward)
            if env.state['hp'][1] >= 2000:
                win1 += 1
            else:
                win2 += 2
            break
    # env.render()
    total_reward *= oom;
    collected_rewards.append(total_reward)
    # print ("total reward", total_reward, "after episode: ", i+1, ". steps: ", j+1)
    # oom = -oom
    oom = om

    print("after " + str(i + 1) + " episodes:")
    
    average = sum(collected_rewards) / num_episodes
    percentage = round(100*(average + 1) / 2, 1)
    score = percentage/100 * (i+1);
    print("average score: ", average)
    print("percentage: ", percentage)
    print("score:", score)
    print()
print("#########")
print("wins1: " + str(win1) + " wins2: " + str(win2))

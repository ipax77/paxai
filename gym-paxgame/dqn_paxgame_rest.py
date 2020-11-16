from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import gym
import gym_paxgame
import os
import requests

import logging
from flask import jsonify, make_response, Flask, request
from flask_restful import abort,Resource, Api
from flask.logging import default_handler

import tensorflow as tf
import numpy as np
from tf_agents.networks import network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.distributions import masked

# RESTurl = "http://localhost:51031/"
# RESTurl = "http://localhost:5000/"
# response = requests.get(url = RESTurl + "paxgame/reset")



tf.compat.v1.enable_v2_behavior()

tempdir = "/data/ml"
policy_dir = os.path.join(tempdir, 'policy')
checkpoint_dir = os.path.join(tempdir, 'checkpoint')

num_iterations = 2 # @param {type:"integer"}

initial_collect_steps = 1  # @param {type:"integer"} 
# collect_steps_per_iteration = 1  # @param {type:"integer"}
collect_steps_per_iteration = 32  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-6  # @param {type:"number"}
log_interval = 1  # @param {type:"integer"}

num_eval_episodes = 1  # @param {type:"integer"}
eval_interval = 1  # @param {type:"integer"}

env_name = 'paxgame-v0'
env = suite_gym.load(env_name)
#env = gym.make(env_name)

env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation)
print('Reward Spec:')
print(env.time_step_spec().reward)
time_step = env.reset()
print('Time step:')
print(time_step)

#action = np.array(1, dtype=np.int32)
#next_time_step, reward1, done, _ = env.step([0, 1, 1])
#next_time_step = env.step(action)
#print('Next time step:')
#print(next_time_step)

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (720,100)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

def observation_and_action_constraint_splitter(observation):
  if (type(observation) is not list):
    return observation, tf.convert_to_tensor(np.ones(545, dtype=np.int32))
  else:
    return observation['state'], tf.convert_to_tensor(np.invert(observation['mask']))

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()    

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

time_step = example_environment.reset()
random_policy.action(time_step)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

compute_avg_return(eval_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

agent.collect_data_spec
agent.collect_data_spec._fields

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)

train_checkpointer.initialize_or_restore()
# saved_policy = tf.compat.v2.saved_model.load(policy_dir)
train_step_counter = tf.compat.v1.train.get_global_step()

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss
  
  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)


app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
api = Api(app)
seperator = ','
class GetMoves(Resource):
    def get(self, request):
        requests = request.split('X')
        moves1 = requests[0].split(seperator)
        moves2 = requests[1].split(seperator)
        round = int(requests[2])
        time_step = example_environment.reset()
        # if (moves1):
        #     for m1 in moves1:
        #         if m1:
        #             env._step((1, int(m1)))
        # if (moves2):
        #     for m2 in moves2:
        #         if m2:
        #             env._step((-1, int(m2)))

        minerals = round * 500
        actions = []
        fs = 0
        while example_environment.state['minerals'][1] < minerals:
            if fs > 200:
                break
            action_step = agent.policy.action(time_step)
            next_time_step = example_environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            actions.append(action_step.action)
            time_step = example_environment.current_time_step()
            fs += 1
        return {'task': seperator.join(str(x) for x in actions)}, 201

api.add_resource(GetMoves, '/getmoves/<string:request>')

if __name__ == '__main__':
     app.run(host='0.0.0.0', port='5077', threaded=True)
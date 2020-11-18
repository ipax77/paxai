from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import abc
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from paxgame_pyenv import PaxGameEnv

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver, tf_policy
from tf_agents.networks import network
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec
from tf_agents.distributions import masked


tf.compat.v1.enable_v2_behavior()

tempdir = "/data/ml"
policy_dir = os.path.join(tempdir, 'policy2')
checkpoint_dir = os.path.join(tempdir, 'checkpoint2')


# utils.validate_py_environment(environment, episodes=5)

num_iterations = 400 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"} 
# collect_steps_per_iteration = 1  # @param {type:"integer"}
collect_steps_per_iteration = 200  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 128  # @param {type:"integer"}
learning_rate = 1e-6  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 200  # @param {type:"integer"}
eval_interval = 200  # @param {type:"integer"}

environment = PaxGameEnv()
time_step = environment.reset()

train_py_env = PaxGameEnv()
eval_py_env = PaxGameEnv()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (720,100)

def observation_and_action_constraint_splitter(observation):
    # return tf.convert_to_tensor(observation['state'])
    # return tf.convert_to_tensor(observation['state']), tf.convert_to_tensor(observation['mask'])
    return observation['state'], observation['mask']


class MaskedQNetwork(network.Network):
  def __init__(self,
               input_tensor_spec,
               action_spec,
               mask_q_value=-100000,
               fc_layer_params=fc_layer_params,
               activation_fn=tf.keras.activations.relu,
               name='MaskedQNetwork'):

      super(MaskedQNetwork, self).__init__(input_tensor_spec, action_spec, name=name)

      self._q_net = q_network.QNetwork(input_tensor_spec['state'], action_spec, fc_layer_params=fc_layer_params,
                                      activation_fn=activation_fn)
      # self._q_net = q_network.QNetwork(input_tensor_spec, action_spec, fc_layer_params=fc_layer_params,
      #                                 activation_fn=activation_fn)
      self._mask_q_value = mask_q_value

  def call(self, observations, step_type, network_state=()):
    state = observations['state']
    mask = observations['mask']

    q_values, _ = self._q_net(state, step_type)

    small_constant = tf.constant(self._mask_q_value, dtype=q_values.dtype, shape=q_values.shape)
    zeros = tf.zeros(shape=mask.shape, dtype=mask.dtype)
    masked_q_values = tf.where(tf.math.equal(zeros, mask),
                               small_constant, q_values)

    return masked_q_values, network_state


# q_net = q_network.QNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     fc_layer_params=fc_layer_params)

q_net = MaskedQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()    

class MaskedRandomPolicy(tf_policy.TFPolicy):
    def __init__(self,
        observation_spec,
        action_spec):

        time_step_spec = ts.time_step_spec(observation_spec)

        super(MaskedRandomPolicy, self).__init__(time_step_spec,action_spec)

    def _action(self, time_step, policy_state, seed):
        if time_step.observation['mask'] is not None:

            mask = time_step.observation['mask']

            zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
            masked_categorical = masked.MaskedCategorical(zero_logits, mask)
            action_ = tf.cast(masked_categorical.sample() + self.action_spec.minimum,
                                self.action_spec.dtype)

            # If the action spec says each action should be shaped (1,), add another
            # dimension so the final shape is (B, 1) rather than (B,).
            if self.action_spec.shape.rank == 1:
                action_ = tf.expand_dims(action_, axis=-1)
        else:
            outer_dims = nest_utils.get_outer_shape(time_step, self._time_step_spec)

            action_ = tensor_spec.sample_spec_nest(
                self._action_spec, seed=seed, outer_dims=outer_dims)

        if time_step is not None:
            with tf.control_dependencies(tf.nest.flatten(time_step)):
                action_ = tf.nest.map_structure(tf.identity, action_)

        policy_info = tensor_spec.sample_spec_nest(self._info_spec)

        if self.emit_log_probability:
            if time_step.observation['mask'] is not None:
                log_probability = masked_categorical.log_prob(
                    action_ - self.action_spec.minimum)
            else:
                _uniform_probability = np.random.uniform(low=0.0, high=1.0)
                action_probability = tf.nest.map_structure(_uniform_probability, self._action_spec)
                log_probability = tf.nest.map_structure(tf.math.log, action_probability)
            policy_info = policy_step.set_log_probability(policy_info, log_probability)

        step = policy_step.PolicyStep(action_, policy_state, policy_info)
        return step

rng_policy = MaskedRandomPolicy(
    train_env.observation_spec(),
    train_env.action_spec()
)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    #time_step = environment.reset()
    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    if traj.is_last:
        return False
    else:
        return True

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    #for _ in range(steps):
    #    collect_step(env, policy, buffer)
    while(collect_step(env, policy, buffer)):
        pass
    # train_py_env.render()

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
# avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
avg_return = compute_avg_return(eval_env, rng_policy, num_eval_episodes)
returns = [avg_return]

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=3,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)

train_checkpointer.initialize_or_restore()
# train_step_counter = tf.compat.v1.train.get_global_step()

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    # collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
    collect_data(train_env, rng_policy, replay_buffer, collect_steps_per_iteration)

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

    if step % 1000 == 0:
        train_checkpointer.save(train_step_counter)
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)
        
train_checkpointer.save(train_step_counter)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)


iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
#plt.ylim(top=250)
plt.show()
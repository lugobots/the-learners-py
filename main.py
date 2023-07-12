# -*- coding: utf-8 -*-
"""DQN Tutorial.ipynb
Code copied from the TensorFlow DQN Tutorial
    https://colab.research.google.com/github/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb

"""
from __future__ import absolute_import, division, print_function

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

import reverb
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment

from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy, policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from tf_agents.specs import tensor_spec

from tf_agents.utils import common

from bot_environment import GameEnvironment
from my_bot import MyBotTrainer, TRAINING_PLAYER_NUMBER
from lugo4py import lugo
from lugo4py.client import LugoClient
from lugo4py.mapper import Mapper
from lugo4py.rl.gym import Gym
from lugo4py.rl.remote_control import RemoteControl
from lugo4py.rl.training_controller import TrainingController

num_iterations = 600000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 2e-2  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 30  # @param {type:"integer"}
eval_interval = 10000  # @param {type:"integer"}

checkpoint_interval = 200
saving_interval = 2000

grpc_address = "localhost:5000"
grpc_insecure = True

stop = threading.Event()


def training(training_ctrl: TrainingController, stop_event: threading.Event):
    print("Starting what actually amtters")
    train_py_env = GameEnvironment(training_ctrl)  # suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    # eval_policy = agent.policy
    # collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            print(f"episode_return: {episode_return}")
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # compute_avg_return(train_env, random_policy, num_eval_episodes)

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2)

    policy_dir = os.path.join(".", 'policy')
    checkpoint_dir = os.path.join("", 'checkpoint')

    global_step = tf.compat.v1.train.get_or_create_global_step()

    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()

    #    agent.collect_data_spec

    #    agent.collect_data_spec._fields

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=None).prefetch(3)

    #    dataset

    iterator = iter(dataset)
    #   print(iterator)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(train_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    for i in range(num_iterations):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if i % checkpoint_interval == 0:
            train_checkpointer.save(global_step)
            print(f'[iteration {i}] checkpoint saved! {i / num_iterations}')

        if step % log_interval == 0:
            print('[iteration {0}] step = {1}: loss = {2}'.format(i, step, train_loss))

        if step % saving_interval == 0:
            print(f'[iteration {i}] policy saved! {i / num_iterations}')
            tf_policy_saver.save(policy_dir)

        if step % eval_interval == 0:
            avg_return = compute_avg_return(train_env, agent.policy, num_eval_episodes)
            print('[iteration {0}] step = {1}: Average Return = {2}'.format(
                i, step, avg_return))
            returns.append(avg_return)

    tf_policy_saver.save(policy_dir)

    # iterations = range(0, num_iterations + 1, eval_interval)
    # plt.plot(iterations, returns)
    # plt.ylabel('Average Return')
    # plt.xlabel('Iterations')
    # plt.ylim(top=250)


if __name__ == '__main__':
    team_side = lugo.TeamSide.HOME
    print('main: Training bot team side = ', team_side)
    # The map will help us see the field in quadrants (called regions) instead of working with coordinates
    # The Mapper will translate the coordinates based on the side the bot is playing on
    mapper = Mapper(20, 10, lugo.TeamSide.HOME)

    # Our bot strategy defines our bot initial position based on its number
    initial_region = mapper.get_region(5, 4)

    # Now we can create the bot. We will use a shortcut to create the client from the config, but we could use the
    # client constructor as well
    lugo_client = LugoClient(
        grpc_address,
        grpc_insecure,
        "",
        team_side,
        TRAINING_PLAYER_NUMBER,
        initial_region.get_center()
    )
    # The RemoteControl is a gRPC client that will connect to the Game Server and change the element positions
    rc = RemoteControl()
    rc.connect(grpc_address)  # Pass address here

    bot = MyBotTrainer(rc)

    gym_executor = ThreadPoolExecutor()
    # Now we can create the Gym, which will control all async work and allow us to focus on the learning part
    gym = Gym(gym_executor, rc, bot, training, {"debugging_log": False})

    players_executor = ThreadPoolExecutor(22)
    gym.with_zombie_players(grpc_address).start(lugo_client, players_executor)


    def signal_handler(_, __):
        print("Stop requested\n")
        lugo_client.stop()
        gym.stop()
        players_executor.shutdown(wait=True)
        gym_executor.shutdown(wait=True)


    signal.signal(signal.SIGINT, signal_handler)

    stop.wait()

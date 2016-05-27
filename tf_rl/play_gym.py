import gym
import tensorflow as tf
import numpy as np
import tf_rl.models as mdls
import time
from tf_rl.controller import ContinuousDeepQ

def set_rand_seeds():
    seed_init = 133
    random.seed(seed_init)
    np.random.seed(seed_init)
    tf.set_random_seed(seed_init)

def run_cartpole():
    env = gym.make('Pendulum-v0')
    ob_space = env.observation_space
    act_space = env.action_space
    observation_size = ob_space.shape[0]
    action_size = act_space.shape[0]

    tf.reset_default_graph()
    #session = tf.InteractiveSession()
    session = tf.Session()

    #actor = mdls.MLP([observation_size, ], [400, 300, action_size], [tf.nn.relu, tf.nn.relu, tf.tanh], scope="actor")
    #critic = mdls.MLP([observation_size, action_size], [400, 300, 1], [tf.nn.relu, tf.nn.relu, tf.identity], scope="critic")

    actor = mdls.MLP([observation_size, ], [64, 64, action_size], [tf.nn.relu, tf.nn.relu, tf.tanh], scope="actor")
    critic = mdls.MLP([observation_size, action_size], [64, 64, 1], [tf.nn.relu, tf.nn.relu, tf.identity], scope="critic")

    #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    #critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    timestr = time.strftime("-%H%M%S")
    writer = tf.train.SummaryWriter("/tmp/test_tb_logs/run" + timestr)

    contDeepQ = ContinuousDeepQ(observation_size, action_size, actor, critic, optimizer, critic_optimizer, session,
                                minibatch_size=32, discount_rate=0.99, max_experience=1000, summary_writer=writer)

    session.run(tf.initialize_all_variables())
    contDeepQ.startup()

    writer.add_graph(session.graph_def)

    #env.monitor.start('/tmp/pendulum-experiment/run_' + timestr)
    for i in range(1500):
        run_epoch(contDeepQ, env)

    #env.monitor.close()


def run_epoch(contDeepQ, env):
    last_observation = env.reset()
    #env.render()

    last_action = None

    for i in range(200):
        new_action = contDeepQ.action(last_observation)
        new_observation, reward, done, _info = env.step(new_action*2.0)
        if done:
            return

        #env.render()

        if last_action is not None:
            contDeepQ.store(last_observation, last_action, reward, new_observation)

        contDeepQ.training_step()
        last_observation = new_observation
        last_action = new_action


if __name__ == '__main__':
    run_cartpole()

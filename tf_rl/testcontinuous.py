#!/usr/bin/env python
import math
import numpy as np
import random
import tensorflow as tf
from tf_rl.controller import ContinuousDeepQ
import time

class SimpleActorMLP(object):
    def __init__(self, scope):
        self.scope = scope

        with tf.variable_scope(self.scope):
            W_initializer =  tf.random_uniform_initializer(-0.003, 0.003) #-1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
            self.W1_var = tf.get_variable("W_1", (1, 1), initializer=W_initializer)
            #self.W1_var = tf.get_variable("W_1", initializer=tf.constant([[3.7]]))
            self.b1 = tf.get_variable("b_1",  (1,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        with tf.variable_scope(self.scope):
            return tf.matmul(xs, self.W1_var) + self.b1

    def variables(self):
        return [self.W1_var, self.b1]

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        return SimpleActorMLP(scope)


class SimpleCriticMLP(object):
    def __init__(self, scope):
        self.scope = scope

        with tf.variable_scope(self.scope):
            W_initializer =  tf.random_uniform_initializer(-0.003, 0.003) #-1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
            self.W1_var = tf.get_variable("W_1", (2, 1), initializer=W_initializer)
            #self.W1_var = tf.get_variable("W_1", initializer=tf.constant([[3.7], [-1.0]]))#W_initializer)
            self.b1 = tf.get_variable("b_1",  (1,), initializer=tf.constant_initializer(0))

            #self.W2_var = tf.get_variable("W_2", (1, 1), initializer=W_initializer)
            #self.W2_var = tf.get_variable("W_2", initializer=tf.constant([[-1.0]]))
            #self.b2 = tf.get_variable("b_2",  (1,), initializer=tf.constant_initializer(0))


    def __call__(self, xs):
        if type(xs) == list:
            xs = tf.concat(1, [xs[0], xs[1]])
        with tf.variable_scope(self.scope):
            return -tf.square(tf.matmul(xs, self.W1_var) + self.b1, "squaring")
            #return tf.matmul(tf.square(tf.matmul(xs, self.W1_var) + self.b1, "squaring"), self.W2_var) + self.b2

    def variables(self):
        return [self.W1_var, self.b1]
        #return [self.W1_var, self.b1, self.W2_var, self.b2]

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        return SimpleCriticMLP(scope)


def learnCritic():
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    critic = SimpleCriticMLP("critic")

    optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)

    input = tf.placeholder(tf.float32, (None, 2), name="input")
    target = tf.placeholder(tf.float32, (None, 1), name="target")

    estimate = critic(input)

    loss = tf.reduce_mean(tf.square(estimate - target))
    train = optimizer.minimize(loss)

    w_hist = tf.histogram_summary("weights1", critic.W1_var)
    b_hist = tf.histogram_summary("biases1", critic.b1)
    w_hist2 = tf.histogram_summary("weights2", critic.W2_var)
    b_hist2 = tf.histogram_summary("biases2", critic.b2)
    accuracy_summary = tf.scalar_summary("accuracy", loss)

    merged = tf.merge_all_summaries()
    timestr = time.strftime("-%H%M%S")
    writer = tf.train.SummaryWriter("/tmp/test_tb_logs/run" + timestr, session.graph_def)

    session.run(tf.initialize_all_variables())

    for i in range(1000000):
        x_data = np.random.rand(100, 2)  # Random input
        y_data = np.square(np.dot(x_data, np.array([[3.700], [-1.00]])))
        feed = {input: x_data, target: y_data}
        if i % 20 == 0:
            foo = session.run([merged, loss], feed_dict=feed)
            writer.add_summary(foo[0], i)
        else:
            foo = session.run(train, feed_dict=feed)

def one_step(contDeepQ):
     # bach states
    states         = np.empty((1, 1))
    newstates      = np.empty((1, 1))
    actions        = np.zeros((1, 1))

    newstates_mask = np.empty((1,))
    rewards        = np.empty((1,))

    states[0] = np.array([2.3])
    newstates[0] = np.array([0.8])
    actions[0] = np.array([0.5])
    rewards[0] = -math.pow(3.7 * 2.3 - 0.5, 2)
    newstates_mask[0] = 1

    contDeepQ.run_learn(states, newstates, newstates_mask, actions, rewards)


def do_rl():
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    #seed_init = 123 #works
    seed_init = 133
    #seed_init = 125  # works!
    random.seed(seed_init)
    np.random.seed(seed_init)
    tf.set_random_seed(seed_init)
    critic = SimpleCriticMLP("critic")
    actor = SimpleActorMLP("actor")

    #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
    #critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
    critic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)

    timestr = time.strftime("-%H%M%S")
    writer = tf.train.SummaryWriter("/tmp/test_tb_logs/run" + timestr)

    contDeepQ = ContinuousDeepQ(1, 1, actor, critic, optimizer, critic_optimizer, session, store_every_nth=1, summary_writer=writer)

    session.run(tf.initialize_all_variables())
    contDeepQ.startup()

    writer.add_graph(session.graph_def)
    #one_step(contDeepQ)

    last_observation = None
    last_action = None
    reward = 0.0
    new_observation = None

    for i in range(1000):
        new_observation = np.random.rand(1)  # Random input
        #new_observation = np.array([random.random()])
        if last_observation is not None:
            #contDeepQ.run_learn(states, newstates, newstates_mask, actions, rewards)
            contDeepQ.store(last_observation, last_action, reward, new_observation)

        new_action = contDeepQ.action(new_observation)
        reward = -math.pow(3.7 * new_observation[0] - new_action, 2)
        contDeepQ.training_step()
        last_observation = new_observation
        last_action = new_action


if __name__ == '__main__':
    do_rl()
    #learnCritic()





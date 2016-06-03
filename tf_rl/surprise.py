import random
import numpy as np
import tensorflow as tf

def run_model():
    config_proto = tf.ConfigProto()
    config_proto.intra_op_parallelism_threads = 1
    config_proto.inter_op_parallelism_threads = 1

    tf.reset_default_graph()
    session = tf.Session(config=config_proto)

    inputs = tf.placeholder(tf.float32, [None, 1], name="inputs")
    target = tf.placeholder(tf.float32, [None, 1], name="inputs")
    layer1_ws = tf.get_variable("w1", (1, 1))
    outputs = tf.reduce_sum(tf.nn.relu(tf.matmul(inputs, layer1_ws)), reduction_indices=1)

    error = tf.reduce_mean(tf.square(outputs - target))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)

    net_opt = optimizer.minimize(error, var_list=[layer1_ws])

    session.run(tf.initialize_all_variables())

    for i in range(10000):
        xval = np.random.uniform(2, 20, size=(5,1))
        yval = xval * 1.5 #+ np.random.randn(5,1)

        _, errorest, weights, estimate = session.run([net_opt, error, layer1_ws, outputs], {inputs: xval, target: yval})
        print(str(estimate) + " " + str(errorest) + " " + str(weights))




if __name__ == '__main__':
    run_model()

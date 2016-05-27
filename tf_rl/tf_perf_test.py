import time
from collections import defaultdict

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

def run_model(num_iters, num_layers, layer_size=64):
    model = Sequential()
    model.add(Dense(output_dim=layer_size, input_dim=2))
    model.add(Activation("relu"))
    for i in range(num_layers):
        model.add(Dense(output_dim=layer_size))
        model.add(Activation("relu"))

    model.add(Dense(output_dim=1))
    model.add(Activation("tanh"))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    total_time = 0

    for i in range(num_iters):
        x = np.random.rand(64, 2)
        start_time = time.perf_counter()
        y = model.predict(x)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    return (total_time/num_iters)

def run_tf_model(num_iters, num_layers, layer_size=64, do_trace=False):
    import tensorflow as tf

    config_proto = tf.ConfigProto()
    config_proto.intra_op_parallelism_threads = 1
    config_proto.inter_op_parallelism_threads = 1

    tf.reset_default_graph()
    session = tf.Session(config=config_proto)
    layers = []

    actor_in = tf.placeholder(tf.float32, [None, 2], name="observation")

    W_var = tf.get_variable("W_0", (2, layer_size))
    b = tf.get_variable("b_0", (layer_size,), initializer=tf.constant_initializer(0))

    hidden_out = tf.nn.relu(tf.matmul(actor_in, W_var) + b)
    layers.append(hidden_out)

    for i in range(num_layers):
        W_name = "W_%d" % (i+1,)
        b_name = "b_%d" % (i+1,)
        W_var = tf.get_variable(W_name, (layer_size, layer_size))
        b = tf.get_variable(b_name, (layer_size,), initializer=tf.constant_initializer(0))
        hidden_out = tf.nn.relu(tf.matmul(hidden_out, W_var) + b)
        layers.append(hidden_out)

    W_var = tf.get_variable("W_last", (layer_size, 1))
    b = tf.get_variable("b_last", (layer_size,), initializer=tf.constant_initializer(0))
    actor_out = tf.nn.tanh(tf.matmul(hidden_out, W_var) + b)

    run_options = None
    run_metadata = None

    if do_trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    session.run(tf.initialize_all_variables())

    timing_map = defaultdict(int)
    total_time = 0

    for i in range(num_iters):
        x = np.random.rand(64, 2)
        if run_options is not None:
            run_metadata = tf.RunMetadata()

        start_time = time.perf_counter()

        session.run([actor_out], {actor_in: x}, options=run_options, run_metadata=run_metadata)

        if run_metadata is not None:
            accumulate_times(run_metadata.step_stats, timing_map)

        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        if i > 0 and i % 1000 == 0 and run_metadata is not None:
            write_timelines(i, run_metadata.step_stats)

    sorted_times =  ((k, timing_map[k]) for k in sorted(timing_map, key=timing_map.get, reverse=True))
    for k,v in sorted_times:
        print("%s %d micros" % (k, v/num_iters))

    return (total_time/num_iters)


def write_timelines(i, step_stats):
    from tensorflow.python.client import timeline
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()
    f = open("tf_trace_" + str(i) + ".json", 'w')
    f2 = open("tf_trace_raw_" + str(i) + ".json", 'w')
    f.write(ctf)
    f2.write(str(step_stats))
    f2.close()
    f.close()

def accumulate_times(step_stats, timing_map):
    for dev_stats in step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
            op_time = node_stats.op_end_rel_micros - node_stats.op_start_rel_micros
            timing_map[node_stats.node_name] += op_time


if __name__ == '__main__':
    for num_layers in range(2, 10, 2):
        for layer_size in range(64, 700, 64):
            avg_time = run_model(10000, num_layers, layer_size)
            #avg_time = run_tf_model(10000, num_layers, layer_size, False)
            print("%d,%d,%f" % (num_layers, layer_size, avg_time))


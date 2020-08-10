import tensorflow as tf
from tensorflow.python.client import timeline

with tf.device('/gpu:0'):
    x = tf.ones([200, 32, 40, 3, 3])
    y = tf.ones([200, 32, 40, 3, 3])
    res = tf.matmul(x, y)

# Run the graph with full trace option
with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_hahaha.json', 'w') as f:
        f.write(ctf)


'''
import tensorflow as tf
import timeit


with tf.device('/gpu:0'):
    mlp1_a = tf.random.normal([200, 32, 1, 256])
    mlp1_b = tf.random.normal([200, 32, 256, 128])

    mlp1_c = tf.random.normal([200, 32, 1, 128])
    mlp1_d = tf.random.normal([200, 32, 128, 32])

    mlp2_a = tf.random.normal([200, 32, 1, 33])
    mlp2_b = tf.random.normal([200, 32, 33, 128])

    mlp2_c = tf.random.normal([200, 32, 1, 128])
    mlp2_d = tf.random.normal([200, 32, 128, 327])

    matmul_a = tf.random.normal([200, 32, 40, 3, 3])
    matmul_b = tf.random.normal([200, 32, 40, 3, 3])
print(mlp1_a.device, mlp1_b.device)


def mlp1_l1_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(mlp1_a, mlp1_b)
    return c


def mlp1_l2_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(mlp1_c, mlp1_d)
    return c


def mlp2_l1_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(mlp2_a, mlp2_b)
    return c


def mlp2_l2_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(mlp2_c, mlp2_d)
    return c


def matmul_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(matmul_a, matmul_b)
    return c


mlp1_l1_time = timeit.timeit(mlp1_l1_run, number=128)
print('warmup:', mlp1_l1_time)

mlp1_l1_time = timeit.timeit(mlp1_l1_run, number=128)
mlp1_l2_time = timeit.timeit(mlp1_l2_run, number=1)
mlp2_l1_time = timeit.timeit(mlp2_l1_run, number=128)
mlp2_l2_time = timeit.timeit(mlp2_l2_run, number=1)
matmul_time = timeit.timeit(matmul_run, number=1)
print('run_time:', mlp1_l1_time, mlp1_l2_time, mlp2_l1_time, mlp2_l2_time, matmul_time)
'''

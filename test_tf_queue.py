import numpy as np
import tensorflow as tf
import time
import os

with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    a = tf.get_variable('v', shape=[2, 2], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('v', shape=[2, 2], initializer=tf.contrib.layers.xavier_initializer())
print(tf.GraphKeys.GLOBAL_VARIABLES)
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
exit()

image_paths = [k for k in os.listdir('.') if k.endswith('.png')]
image_paths_tf = tf.constant(image_paths)
# images = np.arange(50, dtype=np.float32).reshape([50, 1])
# images = np.zeros((50,300, 300, 3), dtype=np.float32)

# def decode_img(idx):
#     def decode_i(idx_):
#         time.sleep(1)
#         return images[idx_]
#     img = tf.py_func(decode_i, [idx], [tf.float32])[0]
#     img.set_shape([1])
#     return img

length = len(image_paths)
queue = tf.train.range_input_producer(length, shuffle = False, num_epochs=1)
idx = queue.dequeue()

# --------------------------------------
img_path = image_paths_tf[idx]
img = tf.read_file(img_path)
img = tf.image.decode_image(img, 3)
# --------------------------------------

img.set_shape([100, 100, 3])
img = tf.train.batch([img], batch_size=10, num_threads=1)
init = tf.group([tf.local_variables_initializer(),tf.global_variables_initializer()])
sess = tf.train.MonitoredTrainingSession()
sess.run(init)
start = time.time()
for _ in range(10):
    print(sess.run(img)[:, 50])
print("Time:{}s".format(time.time()-start))

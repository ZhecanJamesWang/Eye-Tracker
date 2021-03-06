import argparse
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_data_names, load_batch_from_data
import datetime
import random
from mtcnn.mtcnn import mtcnn_handle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

mtcnn_h = mtcnn_handle()
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M")

dataset_path = "../data/GazeCapture"
train_path = dataset_path + '/' + "train"
val_path = dataset_path + '/' + "validation"
test_path = dataset_path + '/' + "test"

# Image Parameters
img_cols = 64
img_rows = 64
img_ch = 3

# Network Parameters
img_size = 64
n_channel = 3
mask_size = 25

# pathway: eye_left and eye_right
conv1_eye_size = 11
conv1_eye_out = 96
pool1_eye_size = 2
pool1_eye_stride = 2

conv2_eye_size = 5
conv2_eye_out = 256
pool2_eye_size = 2
pool2_eye_stride = 2

conv3_eye_size = 3
conv3_eye_out = 384
pool3_eye_size = 2
pool3_eye_stride = 2

conv4_eye_size = 1
conv4_eye_out = 64
pool4_eye_size = 2
pool4_eye_stride = 2

eye_size = 2 * 2 * 2 * conv4_eye_out

# pathway: face
conv1_face_size = 11
conv1_face_out = 96
pool1_face_size = 2
pool1_face_stride = 2

conv2_face_size = 5
conv2_face_out = 256
pool2_face_size = 2
pool2_face_stride = 2

conv3_face_size = 3
conv3_face_out = 384
pool3_face_size = 2
pool3_face_stride = 2

conv4_face_size = 1
conv4_face_out = 64
pool4_face_size = 2
pool4_face_stride = 2

face_size = 2 * 2 * conv4_face_out

# fc layer
fc_eye_size = 128
fc_face_size = 128
fc_face_mask_size = 256
face_face_mask_size = 128
fc_size = 128
fc2_size = 2

def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255.  # scaling
    data = data - np.mean(data, axis=0)  # normalizing
    return np.reshape(data, shape)


def prepare_data(data):
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left).astype('float32')
    eye_right = normalize(eye_right).astype('float32')
    face = normalize(face).astype('float32')
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]


def shuffle_data(data):
    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)
    for i in range(len(data)):
        data[i] = data[i][idx]
    return data


def next_batch(data, batch_size):
    for i in np.arange(0, data[0].shape[0], batch_size):
        # yield a tuple of the current batched data
        yield [each[i: i + batch_size] for each in data]


class EyeTracker(object):
    def __init__(self):
        # tf Graph input
        self.eye_left = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_left')
        self.eye_right = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_right')
        self.face = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='face')
        self.face_mask = tf.placeholder(tf.float32, [None, mask_size * mask_size], name='face_mask')
        self.y = tf.placeholder(tf.float32, [None, 2], name='pos')
        # Store layers weight & bias
        self.weights = {
            'conv1_eye': tf.get_variable('conv1_eye_w',
                                         shape=(conv1_eye_size, conv1_eye_size, n_channel, conv1_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_eye': tf.get_variable('conv2_eye_w',
                                         shape=(conv2_eye_size, conv2_eye_size, conv1_eye_out, conv2_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_eye': tf.get_variable('conv3_eye_w',
                                         shape=(conv3_eye_size, conv3_eye_size, conv2_eye_out, conv3_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_eye': tf.get_variable('conv4_eye_w',
                                         shape=(conv4_eye_size, conv4_eye_size, conv3_eye_out, conv4_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv1_face': tf.get_variable('conv1_face_w',
                                          shape=(conv1_face_size, conv1_face_size, n_channel, conv1_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_face': tf.get_variable('conv2_face_w',
                                          shape=(conv2_face_size, conv2_face_size, conv1_face_out, conv2_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_face': tf.get_variable('conv3_face_w',
                                          shape=(conv3_face_size, conv3_face_size, conv2_face_out, conv3_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_face': tf.get_variable('conv4_face_w',
                                          shape=(conv4_face_size, conv4_face_size, conv3_face_out, conv4_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'fc_eye': tf.get_variable('fc_eye_w', shape=(eye_size, fc_eye_size),
                                      initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face': tf.get_variable('fc_face_w', shape=(face_size, fc_face_size),
                                       initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face_mask': tf.get_variable('fc_face_mask_w', shape=(mask_size * mask_size, fc_face_mask_size),
                                            initializer=tf.contrib.layers.xavier_initializer()),
            'face_face_mask': tf.get_variable('face_face_mask_w',
                                              shape=(fc_face_size + fc_face_mask_size, face_face_mask_size),
                                              initializer=tf.contrib.layers.xavier_initializer()),
            'fc': tf.get_variable('fc_w', shape=(fc_eye_size + face_face_mask_size, fc_size),
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2_w', shape=(fc_size, fc2_size),
                                   initializer=tf.contrib.layers.xavier_initializer())
        }
        self.biases = {
            'conv1_eye': tf.Variable(tf.constant(0.1, shape=[conv1_eye_out])),
            'conv2_eye': tf.Variable(tf.constant(0.1, shape=[conv2_eye_out])),
            'conv3_eye': tf.Variable(tf.constant(0.1, shape=[conv3_eye_out])),
            'conv4_eye': tf.Variable(tf.constant(0.1, shape=[conv4_eye_out])),
            'conv1_face': tf.Variable(tf.constant(0.1, shape=[conv1_face_out])),
            'conv2_face': tf.Variable(tf.constant(0.1, shape=[conv2_face_out])),
            'conv3_face': tf.Variable(tf.constant(0.1, shape=[conv3_face_out])),
            'conv4_face': tf.Variable(tf.constant(0.1, shape=[conv4_face_out])),
            'fc_eye': tf.Variable(tf.constant(0.1, shape=[fc_eye_size])),
            'fc_face': tf.Variable(tf.constant(0.1, shape=[fc_face_size])),
            'fc_face_mask': tf.Variable(tf.constant(0.1, shape=[fc_face_mask_size])),
            'face_face_mask': tf.Variable(tf.constant(0.1, shape=[face_face_mask_size])),
            'fc': tf.Variable(tf.constant(0.1, shape=[fc_size])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_size]))
        }

        # Construct model
        self.pred = self.itracker_nets(self.eye_left, self.eye_right, self.face, self.face_mask, self.weights,
                                       self.biases)

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k, strides):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                              padding='VALID')

    # Create model
    def itracker_nets(self, eye_left, eye_right, face, face_mask, weights, biases):
        # pathway: left eye
        eye_left = self.conv2d(eye_left, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool1_eye_size, strides=pool1_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool2_eye_size, strides=pool2_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool3_eye_size, strides=pool3_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool4_eye_size, strides=pool4_eye_stride)

        # pathway: right eye
        eye_right = self.conv2d(eye_right, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool1_eye_size, strides=pool1_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool2_eye_size, strides=pool2_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool3_eye_size, strides=pool3_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool4_eye_size, strides=pool4_eye_stride)

        # pathway: face
        face = self.conv2d(face, weights['conv1_face'], biases['conv1_face'], strides=1)
        face = self.maxpool2d(face, k=pool1_face_size, strides=pool1_face_stride)

        face = self.conv2d(face, weights['conv2_face'], biases['conv2_face'], strides=1)
        face = self.maxpool2d(face, k=pool2_face_size, strides=pool2_face_stride)

        face = self.conv2d(face, weights['conv3_face'], biases['conv3_face'], strides=1)
        face = self.maxpool2d(face, k=pool3_face_size, strides=pool3_face_stride)

        face = self.conv2d(face, weights['conv4_face'], biases['conv4_face'], strides=1)
        face = self.maxpool2d(face, k=pool4_face_size, strides=pool4_face_stride)

        # fc layer
        # eye
        eye_left = tf.reshape(eye_left, [-1, int(np.prod(eye_left.get_shape()[1:]))])
        eye_right = tf.reshape(eye_right, [-1, int(np.prod(eye_right.get_shape()[1:]))])
        eye = tf.concat([eye_left, eye_right], 1)
        eye = tf.nn.relu(tf.add(tf.matmul(eye, weights['fc_eye']), biases['fc_eye']))

        # face
        face = tf.reshape(face, [-1, int(np.prod(face.get_shape()[1:]))])
        face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc_face']), biases['fc_face']))

        # face mask
        face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc_face_mask']), biases['fc_face_mask']))

        face_face_mask = tf.concat([face, face_mask], 1)
        face_face_mask = tf.nn.relu(
            tf.add(tf.matmul(face_face_mask, weights['face_face_mask']), biases['face_face_mask']))

        # all
        fc = tf.concat([eye, face_face_mask], 1)
        fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['fc']), biases['fc']))
        out = tf.add(tf.matmul(fc, weights['fc2']), biases['fc2'])
        return out

    def train(self, ckpt, plot_ckpt, lr=1e-3, batch_size=128, max_epoch=1000, min_delta=1e-4, patience=10,
              print_per_epoch=10):

        ifCheck = False

        # limit = 1000
        train_names = load_data_names(train_path)[:500]
        # [:limit]
        val_names = load_data_names(val_path)[:500]
        # [:limit]

        train_num = len(train_names)
        val_num = len(val_names)

        print("train_num: ", train_num)
        print("test_num: ", val_num)

        MaxIters = train_num / batch_size
        n_batches = MaxIters

        val_chunk_size = 100
        MaxTestIters = val_num / val_chunk_size
        val_n_batches = val_chunk_size / batch_size

        print("MaxIters: ", MaxIters)
        print("MaxTestIters: ", MaxTestIters)

        print('Train on %s samples, validate on %s samples' % (train_num, val_num))

        # Define loss and optimizer
        self.cost = tf.losses.mean_squared_error(self.y, self.pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

        # Evaluate model
        self.err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.pred, self.y), axis=1)))
        train_loss_history = []
        train_err_history = []
        val_loss_history = []
        val_err_history = []


        # Create the collection
        tf.get_collection("validation_nodes")

        # Add stuff to the collection.
        tf.add_to_collection("validation_nodes", self.eye_left)
        tf.add_to_collection("validation_nodes", self.eye_right)
        tf.add_to_collection("validation_nodes", self.face)
        tf.add_to_collection("validation_nodes", self.face_mask)
        tf.add_to_collection("validation_nodes", self.pred)
        saver = tf.train.Saver(max_to_keep=0)

        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph

        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter("logs", sess.graph)

            # saver.restore(sess, "./my_model/2018-09-15-21-45/model_3_180_train_error_1.8261857_val_error_2.2103159427642822")

            random.shuffle(train_names)
            random.shuffle(val_names)


            # Keep training until reach max iterations
            for n_epoch in range(1, max_epoch + 1):
                print("vvvvvvvvvvvvvvvvvvv")
                print("n_epoch: ", n_epoch)
                epoch_start = timeit.default_timer()
                iter_start = None

                iterTest = 0

                for iter in range(int(MaxIters)):

                    start = timeit.default_timer()

                    # print ("--------------------------------")
                    # print ("iter: ", iter)
                    train_start = iter * batch_size
                    train_end = (iter + 1) * batch_size

                    batch_train_data = load_batch_from_data(mtcnn_h, train_names, dataset_path, batch_size, img_ch,
                                                            img_cols, img_rows, train_start=train_start,
                                                            train_end=train_end)

                    batch_train_data = prepare_data(batch_train_data)

                    print('Loading and preparing training data: %.1fs' % (timeit.default_timer() - start))
                    start = timeit.default_timer()

                    # Run optimization op (backprop)
                    sess.run(self.optimizer, feed_dict={self.eye_left: batch_train_data[0], \
                                                        self.eye_right: batch_train_data[1],
                                                        self.face: batch_train_data[2], \
                                                        self.face_mask: batch_train_data[3],
                                                        self.y: batch_train_data[4]})
                    train_batch_loss, train_batch_err = sess.run([self.cost, self.err],
                                                                 feed_dict={self.eye_left: batch_train_data[0], \
                                                                            self.eye_right: batch_train_data[1],
                                                                            self.face: batch_train_data[2], \
                                                                            self.face_mask: batch_train_data[3],
                                                                            self.y: batch_train_data[4]})

                    print("train_batch_loss: ", train_batch_loss, "train_batch_err: ", train_batch_err)

                    # train_loss.append(train_batch_loss)
                    # train_err.append(train_batch_err)
                    train_loss_history.append(train_batch_loss)
                    train_err_history.append(train_batch_err)

                    print('Training on batch: %.1fs' % (timeit.default_timer() - start))


                    if iter % 30 == 0:
                        ifCheck = True

                    if ifCheck:

                        start = timeit.default_timer()

                        if iterTest + 1 >= MaxTestIters:
                            iterTest = 0

                        # test_start = iterTest * val_chunk_size
                        # test_end = (iterTest+1) * val_chunk_size
                        test_start = 0
                        test_end = val_chunk_size

                        val_data = load_batch_from_data(mtcnn_h, val_names, dataset_path, val_chunk_size, img_ch,
                                                        img_cols, img_rows, train_start=test_start, train_end=test_end)

                        val_n_batches = val_data[0].shape[0] / batch_size + (val_data[0].shape[0] % batch_size != 0)

                        val_data = prepare_data(val_data)

                        print('Loading and preparing val data: %.1fs' % (timeit.default_timer() - start))
                        start = timeit.default_timer()

                        val_loss = 0.
                        val_err = 0
                        for batch_val_data in next_batch(val_data, batch_size):
                            val_batch_loss, val_batch_err = sess.run([self.cost, self.err],
                                                                     feed_dict={self.eye_left: batch_val_data[0], \
                                                                                self.eye_right: batch_val_data[1],
                                                                                self.face: batch_val_data[2], \
                                                                                self.face_mask: batch_val_data[3],
                                                                                self.y: batch_val_data[4]})
                            val_loss += val_batch_loss / val_n_batches
                            val_err += val_batch_err / val_n_batches

                        print("val_loss: ", val_loss, "val_err: ", val_err)
                        iterTest += 1

                        print('Testing on chunk: %.1fs' % (timeit.default_timer() - start))
                        start = timeit.default_timer()

                        if iter_start:
                            print('batch iters runtime: %.1fs' % (timeit.default_timer() - iter_start))
                        else:
                            iter_start = timeit.default_timer()

                        print("now: ", now)
                        print("learning rate: ", lr)

                        print('Epoch %s/%s Iter %s, train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f' % (n_epoch, max_epoch, iter, np.mean(train_loss_history), np.mean(train_err_history), np.mean(val_loss_history), np.mean(val_err_history)))

                        val_loss_history.append(val_loss)
                        val_err_history.append(val_err)

                        plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_loss_history), np.array(val_err_history), start=0, per=1, save_file=plot_ckpt + "/cumul_loss_" + str(n_epoch) + "_" + str(iter) + ".png")

                        save_path = ckpt + "model_" + str(n_epoch) + "_" + str(iter) + "_train_error_history_%s" % (
                            np.mean(train_err_history)) + "_val_error_history_%s" % (np.mean(val_err_history))

                        save_path = saver.save(sess, save_path)

                        print("Model saved in file: %s" % save_path)

                        ifCheck = False

                        print('Saving models and plotting loss: %.1fs' % (timeit.default_timer() - start))

                print('epoch runtime: %.1fs' % (timeit.default_timer() - epoch_start))

            return train_loss_history, train_err_history, val_loss_history, val_err_history


def extract_validation_handles(session):
    """ Extracts the input and predict_op handles that we use for validation.
    Args:
        session: The session with the loaded graph.
    Returns:
        validation handles.
    """
    # valid_nodes = tf.get_collection_ref("validation_nodes")
    print(tf.get_default_graph().get_all_collection_keys())

    valid_nodes = tf.get_collection("validation_nodes")
    print("len(valid_nodes): ", len(valid_nodes))
    # if len(valid_nodes) != 5:
    #     raise Exception("ERROR: Expected 5 items in validation_nodes, got %d." % len(valid_nodes))
    return valid_nodes


def load_model(session, save_path):
    """ Loads a saved TF model from a file.
    Args:
        session: The tf.Session to use.
        save_path: The save path for the saved session, returned by Saver.save().
    Returns:
        The inputs placehoder and the prediction operation.
    """
    print("Loading model from file '%s'..." % save_path)

    meta_file = save_path + ".meta"
    if not os.path.exists(meta_file):
        raise Exception("ERROR: Expected .meta file '%s', but could not find it." % meta_file)

    saver = tf.train.import_meta_graph(meta_file)
    # It's finicky about the save path.
    save_path = os.path.join("./", save_path)
    saver.restore(session, save_path)

    # Check that we have the handles we expected.
    print("Successfully loaded !!!")
    return extract_validation_handles(session)


def validate_model(sess, val_names, val_ops, plot_ckpt, batch_size=200):
    """ Validates the model stored in a session.
    Args:
        session: The session where the model is loaded.
        val_data: The validation data to use for evaluating the model.
        val_ops: The validation operations.
    Returns:
        The overall validation error for the model. """

    print("Validating model...")
    val_num = len(val_names)
    print("test_num: ", val_num)

    MaxTestIters = int(val_num / batch_size)
    print("MaxTestIters: ", MaxTestIters)

    val_err = []

    iter_start = None

    cum_err_num = 0

    print("len(val_ops): ", len(val_ops))
    eye_left, eye_right, face, face_mask, pred = val_ops
    # eye_left, eye_right, face, face_mask, pred_xy, pred_ang_left, pred_ang_right = val_ops

    y = tf.placeholder(tf.float32, [None, 2], name='pos')
    err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred, y), axis=1)))
    # err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred_xy, y), axis=1)))

    for iterTest in range(MaxTestIters):
        test_start = iterTest * batch_size + cum_err_num
        test_end = test_start + batch_size


        batch_val_data, cum_err = load_batch_from_data(mtcnn_h, val_names, dataset_path, batch_size, img_ch, img_cols, img_rows, val_num, train_start=test_start, train_end=test_end)
        batch_val_data = prepare_data(batch_val_data)

        cum_err_num += cum_err

        val_batch_err = sess.run(err, feed_dict={eye_left: batch_val_data[0], \
                                                 eye_right: batch_val_data[1], face: batch_val_data[2], \
                                                 face_mask: batch_val_data[3], y: batch_val_data[4]})
        print ("val_batch_err: ", val_batch_err)

        val_err.append(val_batch_err)

        if iterTest % 10 == 0:
            print('IterTest %s, val error: %.5f' % \
                  (iterTest, np.mean(val_err)))
            print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            # plot_loss(np.array(train_loss), np.array(train_err), np.array(Val_err), start=0, per=1, save_file=plot_ckpt + "/testing_loss_" + str(n_epoch) + "_" + str(iterTest) + ".png")

            if iter_start:
                print('10 iters runtime: %.1fs' % (timeit.default_timer() - iter_start))
            else:
                iter_start = timeit.default_timer()

    return np.mean(val_err)


def plot_loss(train_loss, train_err, test_loss, test_err, start=0, per=1, save_file='loss.png'):
    print("----plot loss----")

    idx = np.arange(start, len(train_loss), per)
    fig, ax1 = plt.subplots()
    label = 'train loss'
    lns1 = ax1.plot(idx, train_loss[idx], 'b-', alpha=1.0, label='train loss')
    ax1.set_xlabel('epochs')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(lns1, label, loc=0)

    fig.tight_layout()
    plt.savefig(save_file + "_train_loss" + ".png")

    fig, ax2 = plt.subplots()
    label = 'train_err'
    lns2 = ax2.plot(idx, train_err[idx], 'r-', alpha=1.0, label='train_err')
    ax2.set_ylabel('error', color='r')
    ax2.tick_params('y', colors='r')
    ax1.legend(lns2, label, loc=0)

    fig.tight_layout()
    plt.savefig(save_file + "_train_err" + ".png")

    idx = np.arange(start, len(test_loss), per)
    idx_30 = np.arange(start, len(test_loss), per) * 30
    fig, ax1 = plt.subplots()
    label = 'test loss'
    lns3 = ax1.plot(idx_30, test_loss[idx], 'c-', alpha=1.0, label='test loss')
    ax1.set_xlabel('epochs')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(lns3, label, loc=0)

    fig.tight_layout()
    plt.savefig(save_file + "_test_loss" + ".png")

    fig, ax2 = plt.subplots()
    label = 'test_err'
    lns4 = ax2.plot(idx_30, test_err[idx], 'g-', alpha=1.0, label='test_err')
    ax2.set_ylabel('error', color='r')
    ax2.tick_params('y', colors='r')
    ax1.legend(lns4, label, loc=0)

    fig.tight_layout()
    plt.savefig(save_file + "_test_err" + ".png")
    # plt.show()


def train(args):
    start = timeit.default_timer()
    plot_ckpt = "plots/" + date
    if not os.path.exists(plot_ckpt):
        os.makedirs(plot_ckpt)

    out_model = "my_model"
    print("out_model: ", out_model.split())
    ckpt = out_model + "/" + date + "/"
    print("ckpt: ", ckpt)
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)

    et = EyeTracker()
    train_loss_history, train_err_history, val_loss_history, val_err_history = et.train(ckpt, plot_ckpt,
                                                                                        lr=args.learning_rate,
                                                                                        batch_size=args.batch_size,
                                                                                        max_epoch=args.max_epoch, \
                                                                                        min_delta=1e-4, \
                                                                                        patience=args.patience, \
                                                                                        print_per_epoch=args.print_per_epoch)

    print('Total training runtime: %.1fs' % (timeit.default_timer() - start))

    # plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_err_history), start=0, per=1, save_file= plot_ckpt + "/total_loss.png")

    if args.save_loss:
        with open(plot_ckpt + "/" + args.save_loss, 'w') as outfile:
            np.savez(outfile, train_loss_history=train_loss_history, train_err_history=train_err_history, \
                     val_loss_history=val_loss_history, val_err_history=val_err_history)


def test(args):
    print("--------testing---------")
    plot_ckpt = "plots/" + date
    # if not os.path.exists(plot_ckpt):
    #     os.makedirs(plot_ckpt)

    val_names = load_data_names(val_path)
    # [:2000]
    # Load and validate the network.
    with tf.Session() as sess:
        val_ops = load_model(sess, args.load_model)
        error = validate_model(sess, val_names, val_ops, plot_ckpt, batch_size=args.batch_size)
        print('Overall validation error: %f' % error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    # parser.add_argument('-i', '--input', required=True, type=str, help='path to the input data')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=60, help='max number of iterations')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    # 0.0025
    # 0.0001
    # 0.00001
    # 0.000001

    # default = "pretrained_models/itracker_adv/model-23",
    # default ='my_model/2018-09-06-23-11/model_1_1500_train_error_2.3585315_val_error_2.000537872314453'
    # default='my_model/2018-09-07-11-15/model_4_420_train_error_2.2030365_val_error_1.8307928442955017'
    # , default='my_model/2018-09-08-23-32/model_1_25_train_error_2.202213_val_error_2.193189859390259'
    # default = "my_model/pretrained/model_4_1800_train_error_3.5047762_val_error_5.765135765075684"
    parser.add_argument('-bs', '--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('-p', '--patience', type=int, default=np.Inf, help='early stopping patience')
    parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration')
    parser.add_argument('-sm', '--save_model', type=str, default='my_model', help='path to the output model')
    parser.add_argument('-lm', '--load_model', type=str, default = "pretrained_models/2018-08-30-23-11/model_3_630_train_error_3.510948_val_error_4.466572284698486")
    parser.add_argument('-pl', '--plot_loss', type=str, default='loss.png', help='plot loss')
    parser.add_argument('-sl', '--save_loss', type=str, default='loss.npz', help='save loss')
    args = parser.parse_args()

    # if args.train:
    # train(args)
    # else:
    #     if not args.load_model:
    #         raise Exception('load_model arg needed in test phase')
    test(args)


if __name__ == '__main__':
    main()

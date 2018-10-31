import pickle
import numpy as np

[pred_xy_list, y_list] = pickle.load( open( file_name, "rb" ) )

# err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred_xy, y), axis=1)))
err = np.mean(np.sqrt(((pred_xy - y) ** 2).sum()))

print "error: ", err

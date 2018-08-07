from load_data import load_data_names, load_batch_from_names_random
import numpy as np

dataset_path = "GazeCapture"
train_path = "GazeCapture/train"
val_path = "GazeCapture/validation"
test_path = "GazeCapture/test"

# train parameters
# n_epoch = args.max_epoch
batch_size = 16
# patience = args.patience

# image parameter
img_cols = 64
img_rows = 64
img_ch = 3

limit = 100

# train data
train_names = load_data_names(train_path)[:limit]
# validation data
val_names = load_data_names(val_path)[:limit]
# test data
test_names = load_data_names(test_path)[:limit]

# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_ch, img_cols, img_rows):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y


for i in range(100):
    batch = next(generator_train_data(train_names, dataset_path, batch_size, img_ch, img_cols, img_rows))
    print (len(batch[0]))
    print (np.asarray(batch[0][0]).shape)
    print (batch[1].shape)

# print (len(batch))
# print (batch.shape)

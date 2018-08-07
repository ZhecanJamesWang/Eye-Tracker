from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random

dataset_path = "/cvgl/group/GazeCapture/gazecapture"

train_path = "/cvgl/group/GazeCapture/train"
val_path = "/cvgl/group/GazeCapture/validation"
test_path = "/cvgl/group/GazeCapture/test"

# train parameters
n_epoch = args.max_epoch
batch_size = 16
# patience = args.patience

# image parameter
img_cols = 64
img_rows = 64
img_ch = 3


# train data
train_names = load_data_names(train_path)
# validation data
val_names = load_data_names(val_path)
# test data
test_names = load_data_names(test_path)

# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_ch, img_cols, img_rows):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y

generator_train_data(train_names, dataset_path, batch_size, img_ch, img_cols, img_rows)

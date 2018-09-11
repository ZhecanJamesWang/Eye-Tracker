import numpy as np
npzfile = np.load("data/eye_tracker_train_and_val.npz")

print npzfile
print npzfile.keys()

limit = 100

train_eye_left = npzfile["train_eye_left"][:limit]
train_eye_right = npzfile["train_eye_right"][:limit]
train_face = npzfile["train_face"][:limit]
train_face_mask = npzfile["train_face_mask"][:limit]
train_y = npzfile["train_y"][:limit]

for i in range(len(train_y)):
    print train_y[i]

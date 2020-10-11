import numpy as np
import os
import cv2
import pickle
import random


DIRECTORY = r'train'
CATEGORIES = ['real', 'spoof']

data = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_arr = cv2.resize(arr, (100, 100))
        data.append([new_arr, label])


random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))


# predict_data = []
#
# test_path = 'test'
#
# for img in os.listdir(test_path):
#     img_path = os.path.join(test_path, img)
#     arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     new_arr = cv2.resize(arr, (100, 100))
#     predict_data.append(new_arr)
#
# X_predict = np.array(predict_data)

# pickle.dump(X_predict, open('X_predict.pkl', 'wb'))

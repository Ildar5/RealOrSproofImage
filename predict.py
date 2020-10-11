import cv2
import numpy as np
import os
import tensorflow


CATEGORIES = ['real', 'spoof']

loaded_model = tensorflow.keras.models.load_model('softmax_model')
loaded_model.load_weights('softmax_model_w')

loaded_model.compile(
    optimizer = 'adam',
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)


test_path = 'test'
result = open("result.txt", "a")


test_data = []

for img in os.listdir(test_path):
    img_path = os.path.join(test_path, img)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(image, (100, 100))
    new_arr = np.array(new_arr)
    new_arr = new_arr / 255.0
    new_arr = new_arr.reshape(-1, 100, 100, 1)
    test_data.append(new_arr)

    prediction = loaded_model.predict([new_arr], batch_size=128)

    result.write("{0}, {1}" . format(img, prediction[0][0]))
    result.write('\n')
    # print("{0} {1} {2}" . format(CATEGORIES[prediction.argmax()], prediction[0][0], img))

result.close()

print('finish')




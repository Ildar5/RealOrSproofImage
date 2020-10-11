from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import preprocessing
import pickle


model = Sequential()

model.add(Conv2D(128, 3, activation = 'relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(
    optimizer = 'adam',
    loss = SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X / 255
X = X.reshape(-1, 100, 100, 1)


model.fit(X, y, epochs = 15, validation_split = 0.1, batch_size = 128)

model.save('softmax_model')
model.save_weights('softmax_model_w')






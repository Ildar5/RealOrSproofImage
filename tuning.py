from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D
from kerastuner.tuners import RandomSearch
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pickle
from sklearn.model_selection import train_test_split


X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X / 255
X = X.reshape(-1, 100, 100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 10)


def search_model(hp):

  model = Sequential()

  for i in range(hp.Int('Conv Layer', min_value = 1, max_value = 4)):
    model.add(Conv2D(hp.Choice(f"Conv2D_{i}_filters", [32, 64, 128, 256]), 3, activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())

  model.add(Dense(128, activation = 'relu'))
  model.add(Dense(2, activation = 'softmax'))

  model.compile(
      optimizer = 'adam',
      loss = SparseCategoricalCrossentropy(),
      metrics = ['accuracy']
  )

  return model


tuner = RandomSearch(
    search_model,
    objective = 'val_accuracy',
    max_trials = 32,
    directory = 'best_model_1'
)

tuner.search(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size = 32)
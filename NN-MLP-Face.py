import numpy as np
from databaseUtil import load_database
from mathUtil import shuffle_in_unison
from plotUtil import plot_history, plot_show

import keras
from keras.utils.np_utils import to_categorical

train_path = "./Data/Face-ORL-Train.dbs"
test_path = "./Data/Face-ORL-Test.dbs"
train, train_label = load_database(train_path)
test, test_label = load_database(test_path)

shuffle_in_unison(train, train_label)
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

num_epochs = 24
model = keras.models.Sequential()
model.add(keras.layers.Dense(40, activation="sigmoid", input_shape=(270,)))
# model.add(keras.layers.Dense(20, activation="sigmoid")) 
model.add(keras.layers.Dense(40, activation="sigmoid"))

opt = keras.optimizers.RMSprop(learning_rate=0.01, decay=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(train, train_label, batch_size=None, epochs=num_epochs)

loss, acc = model.evaluate(train, train_label)
print('Train Loss     :: ', loss)
print('Train Accuracy :: ', acc)
loss, acc = model.evaluate(test, test_label)
print('Test Loss     :: ', loss)
print('Test Accuracy :: ', acc)

plot_history(history)
plot_show()

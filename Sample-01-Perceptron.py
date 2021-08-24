import sys
from os import getcwd
from os.path import exists, join

dir = getcwd()
if not exists(join(dir, 'Include')):
    dir = dir[:dir.rfind('\\')]
if exists(join(dir, 'Include')):
    sys.path.insert(1, join(dir, 'Include'))

from databaseUtil import load_database
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

data, label = load_database('Data/Points-25.dbs')
plt.scatter(data[:, 0], data[:, 1], c=label)

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_shape=(2, )))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(data, label, epochs=10)
model.evaluate(data)

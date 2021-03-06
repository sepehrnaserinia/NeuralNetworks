from databaseUtil import load_database_data
from mathUtil import shuffle_in_unison
from plotUtil import plot_history, plot_history_val
import numpy as np
import matplotlib.pyplot as plt

data_path = "../Database/Digits/"
train, train_label, test, test_label = load_database_data(data_path=data_path, img_output_size=64)

epochs = int(input("Enter Number of Epochs :: "))
val_check = input("Use Validation?? (Y/N) :: ")

import keras
from keras.utils import to_categorical
from keras import models, layers, optimizers

train = np.expand_dims(train, axis=3)
test = np.expand_dims(test, axis=3)
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
shuffle_in_unison(train, train_label)

model = models.Sequential()
model.add(layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=(64, 64, 1,)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

if val_check is 'y':
    val_step = train.shape[0] // 5
    train_val = train[:val_step]
    train_part = train[val_step:]

    train_label_val = train_label[:val_step]
    train_label_part = train_label[val_step:]

    network_history = model.fit(train_part, train_label_part, 
                                epochs=epochs, batch_size=128, 
                                validation_data=(train_val, train_label_val))
else :
    network_history = model.fit(train, train_label, epochs=epochs, batch_size=128)

test_loss, test_acc = model.evaluate(test, test_label)
print("Test Loss     :: {0}".format(test_loss))
print("Test Accuracy :: {0}".format(test_acc))

if val_check is 'y':
    plot_history_val(network_history)
else :
    plot_history(network_history)
plt.ylim(0, 1.1)
plt.show()
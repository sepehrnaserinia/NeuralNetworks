from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import mathUtil as mu

def plot_show():
    plt.show()

def plot_history(history):
    plt.figure()
    acc = history.history['accuracy']
    loss = history.history['loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'red', label='Training Loss')
    plt.plot(epochs, acc, 'blue', label='Training Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.grid(True)

def plot_history_val(history):
    plt.figure()
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'red', label='Train Loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
    plt.plot(epochs, acc, 'blue', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'cyan', label='Validation Accuracy')
    plt.title('Train and Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.grid(True)

def plot_history_matrix(history):
    plt.figure()
    loss = history[0]
    acc = history[1]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'red', label='Training Loss')
    plt.plot(epochs, acc, 'blue', label='Training Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.grid(True)

def plot_decision_boundary_simple(min, max, centers, colors):
    x_min, x_max = min, max
    y_min, y_max = min, max
    step = (max - min) / 100
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    points = np.c_[xx.ravel(), yy.ravel()]  
    
    Z = []
    for point in points:
        min_dis = (max - min) * sqrt(2)
        for center, nclass in zip(centers, range(3)):
            dis = mu.distance([point, center])
            if dis < min_dis:
                min_dis = dis
                nclass_win = nclass
        Z.append(nclass_win)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    plt.figure()    
    plt.contourf(xx, yy, Z, cmap='Spectral')

def plot_decision_boundary(model, data, label):
    # Set min and max values and give it some padding
    x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    step = 0.05
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    points = np.c_[xx.ravel(), yy.ravel()]  

    Z = model.predict_classes(points)
    Z = Z.reshape(xx.shape)
    plt.figure()    
    plt.contourf(xx, yy, Z, cmap='Spectral')
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap='Spectral')
    
    if label.max() == 1:
        Z_Soft = model.predict(points)
        Z_Soft = Z_Soft.reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, Z_Soft, cmap='Spectral')
        plt.scatter(data[:, 0], data[:, 1], c=label, cmap='Spectral')

if __name__ == "__main__":
    print("Header File...")

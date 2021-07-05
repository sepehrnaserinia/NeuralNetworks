import numpy as np
import matplotlib.pyplot as plt
from fileUtil import read_matrix


data_path = "../Database/_Database/"

file = open(data_path + "Regression.dbs", 'r')
x = np.transpose(read_matrix(file))
y = np.transpose(read_matrix(file))
noisySig = np.transpose(read_matrix(file))
file.close()

file = open(data_path + "Regression-Output.smat", 'r')
predict_x = np.transpose(read_matrix(file))
predict_y = np.transpose(read_matrix(file))
file.close()

fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.grid(linestyle='--', linewidth=1)
ax1.plot(x, y, label='Actual function, not noisy', linewidth=1.0, c='black')
ax1.plot(x, noisySig, 'g.', label='Raw noisy input data')
ax1.plot(predict_x, predict_y, label='Output of the Neural Network', linewidth=2.0, c='red')
plt.legend()
plt.show()


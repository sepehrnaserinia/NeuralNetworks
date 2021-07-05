import matplotlib.pyplot as plt
import os
from os.path import join
from plotUtil import plot_history_matrix
from fileUtil import read_matrix  
    
path_output = "../Database/_Output/"
filename = "History.smat"

file = open(join(path_output, filename), 'r')
history = read_matrix(file)
file.close()

plot_history_matrix(history)
plt.show()
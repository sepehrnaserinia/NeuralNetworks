import numpy as np

def read_matrix(file):
    line = file.readline()
    index1 = line.find('(') + 1
    index2 = line.find(', ')
    rows = int(line[index1:index2])
    index1 = line.find(', ') + 2
    index2 = line.find(')')
    cols = int(line[index1:index2])

    dataMatrix = np.zeros((rows, cols))
    for i in range(rows):
        line = file.readline()
        data = line.split(', ')[:cols]
        dataMatrix[i] = data
    return dataMatrix

if __name__ == "__main__":
    print("Header File...")

import numpy as np
import pandas as pd

# can run in command line with these arguments: python3 FeatureExtraction.py trainDataSet.csv dummy.csv
from numpy.core._multiarray_umath import ndarray

output = np.zeros((1000, 11))


# def features - further explanation in features.txt
def feature1(r):  # bottom left corner
    return int(r[0])


def feature2(r):  # center column
    player1 = 0
    player2 = 0
    for i in r:
        i = int(i)
        if r[i + 3] == '1':
            player1 += 1
        if r[i + 3] == '2':
            player2 += 1
        i += 7
    if player1 > player2:
        return int(1)
    if player2 > player1:
        return int(2)
    if player2 == player1:
        return int(0)


def feature3(r):  # center-1 column
    player1 = 0
    player2 = 0
    for i in r:
        i = int(i)
        if r[i + 2] == '1':
            player1 += 1
        if r[i + 2] == '2':
            player2 += 1
        i += 7
    if player1 > player2:
        return int(1)
    if player2 > player1:
        return int(2)
    else:
        return int(0)


def feature4(r):  # center+1 column
    player1 = 0
    player2 = 0
    for i in r:
        i = int(i)
        if r[i + 4] == '1':
            player1 += 1
        if r[i + 4] == '2':
            player2 += 1
        i += 7
    if player1 > player2:
        return int(1)
    if player2 > player1:
        return int(2)
    else:
        return int(0)


def feature5(r):  # bottom row
    player1 = 0
    player2 = 0
    i = 0
    while i < 7:
        if r[i] == '1':
            player1 += 1
        if r[i] == '2':
            player2 += 1
        i += 1
    if player1 > player2:
        return int(1)
    if player2 > player1:
        return int(2)
    else:
        return int(0)


def feature6(r):   # center piece bottom
    return int(r[3+7*3])


def feature7(r):   # center piece top
    return int(r[3+7*4])


def feature8(r):
    player1 = 0
    player2 = 0
    for i in r:
        i = int(i)
        if r[i + 2] == '1':
            player1 += 1
        if r[i + 2] == '2':
            player2 += 1
        if r[i + 3] == '1':
            player1 += 1
        if r[i + 3] == '2':
            player2 += 1
        if r[i + 4] == '1':
            player1 += 1
        if r[i + 4] == '2':
            player2 += 1
        i += 7
    if player1 > player2:
        return int(1)
    if player2 > player1:
        return int(2)
    else:
        return int(0)


def getLabels(r):
    return int(r[42])


def getFeatures(loadInput):
    j = 0  # j is equal to index of current row
    for row in loadInput:    # actual function here to put features into array
        output[j, 1] = feature1(row)
        output[j, 2] = feature2(row)
        output[j, 3] = feature3(row)
        output[j, 4] = feature4(row)
        output[j, 5] = feature5(row)
        output[j, 6] = feature8(row)  # control all three center columns
        if output[j, 3] == output[j, 4] or output[j, 3] == output[j, 5]:  # control center and neighbor column
            output[j, 7] = output[j, 3]
        output[j, 8] = feature6(row)
        output[j, 9] = feature7(row)
        if output[j, 8] == output[j, 9]:  # control both center pieces in middle column
            output[j, 0] = output[j, 8]
        output[j, 10] = getLabels(row)
        j += 1

    np.savetxt('features.csv', output, delimiter=",", fmt="%1i")
    return pd.read_csv('features.csv', dtype='int', header=None)
    # return np.loadtxt('features.csv', delimiter=',', dtype='int')

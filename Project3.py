import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

if len(sys.argv) != 3:
    sys.exit("Must specify input and output files")
firstInputCSV = sys.argv[1]
secondInputCSV = sys.argv[2]

# inputReader = csv.reader(open(firstInputCSV, 'r'))
loadInput = np.loadtxt(firstInputCSV, delimiter=',', dtype='str')

encodedData = np.zeros((42001, 3))
counter = 0
offset = 0
for row in loadInput:
    if counter > 0:
        for i in range (len(row)):
            encodedData[i+offset][int(row[i])] = 1
        offset += 42
    counter += 1

# actual coding stuff here

# changes = [
#     ['1 dozen'],
#     ['1 banana'],
#     # ['1 dollar', 'elephant', 'heffalump'],
# ]
# print(changes)
#
# outputFilePath = "./" + str(secondInputCSV)
# with open(outputFilePath, 'w', newline='') as outputFile:
#     # writer = csv.writer(outputFile, delimiter=',')
#     df = pd.read_csv(firstInputCSV)
#     df['Hola'] = 'hola'
#     df.to_csv(outputFile, index=False)
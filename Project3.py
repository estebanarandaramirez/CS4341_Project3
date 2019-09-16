import csv
import sys
import pandas as pd
import numpy as np
from FeatureExtraction import getFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics


def DecisionTree(X, Y):
    decisionTree = DecisionTreeClassifier(criterion='entropy')
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        decisionTree = decisionTree.fit(X[train], Y[train])
        Y_pred = decisionTree.predict(X[test])
        cvscores.append(metrics.accuracy_score(Y[test], Y_pred) * 100)
    print("DTree - Mean(+/-Standard Deviation): %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("DTree - Final test accuracy: %.2f%%" % (metrics.accuracy_score(Y[test], Y_pred) * 100))
    with open('DTresults.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow((' ', np.mean(cvscores), np.std(cvscores), metrics.accuracy_score(Y[test], Y_pred) * 100))
    with open(outputFilePath, 'a', newline='') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(Y_pred)


def RandomForest(X, Y):
    rTree = RandomForestClassifier(criterion='entropy', n_estimators=10)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        rTree = rTree.fit(X[train], Y[train].ravel())
        Y_pred = rTree.predict(X[test])
        cvscores.append(metrics.accuracy_score(Y[test], Y_pred) * 100)
    print("RForest - Mean(+/-Standard Deviation): %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("RForest - Final test accuracy: %.2f%%" % (metrics.accuracy_score(Y[test], Y_pred) * 100))
    with open('RFresults.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow((' ', np.mean(cvscores), np.std(cvscores), metrics.accuracy_score(Y[test], Y_pred) * 100))


def PrintYTest(X, Y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in kfold.split(X, Y):
        break
    with open(outputFilePath, 'w', newline='') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(Y[test])


seed = np.random.seed(1)

if len(sys.argv) != 3:
    sys.exit("Must specify input and output files")

firstInputCSV = sys.argv[1]
loadInput = np.loadtxt(firstInputCSV, delimiter=',', dtype='str', skiprows=1)

secondInputCSV = sys.argv[2]
outputFilePath = "./" + str(secondInputCSV)

with open('DTresults.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow((' ', 'Mean', 'STD', 'Final test accuracy'))
with open('RFresults.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow((' ', 'Mean', 'STD', 'Final test accuracy'))

features = getFeatures(loadInput)
Y = features[10].values.reshape(-1, 1)
all = features[[0,1,2,3,4,5,6,7,8,9]]
X = all.values.reshape(-1, 10)
PrintYTest(X, Y)

for i in range (10):
    X = features[i].values.reshape(-1, 1)
    with open('DTresults.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature %i:' % (i)])
    with open('RFresults.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature %i:' % (i)])
    print("Feature #%i:" % (i))
    DecisionTree(X, Y)
    RandomForest(X, Y)

print()

with open('DTresults.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['All features:'])
with open('RFresults.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['All features:'])
print("All features combined:")
all = features[[0,1,2,3,4,5,6,7,8,9]]
X = all.values.reshape(-1, 10)
DecisionTree(X, Y)
RandomForest(X, Y)

print()

for i in range (10):
    featuresToTest = features.loc[:, features.columns != i]
    X = features[i].values.reshape(-1, 1)
    with open('DTresults.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['All except %i:' % (i)])
    with open('RFresults.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['All except %i:' % (i)])
    print("All features except #%i:" % (i))
    DecisionTree(X, Y)
    RandomForest(X, Y)
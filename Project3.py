import csv
import sys
import pandas as pd
import numpy as np
from FeatureExtraction import  getFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn import metrics

seed = np.random.seed(1)

def DecisionTree(X, Y, i):
    # decisionTree = DecisionTreeClassifier()
    # scores = cross_val_score(estimator=decisionTree, X=X, y=Y, cv=5)
    # print("Accuracy for feature %i: Mean %f, STD %f" % (i+1, scores.mean(), scores.std()))
    # return scores.mean(), scores.std()

    decisionTree = DecisionTreeClassifier(criterion='entropy')
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    i = 1
    for train, test in kfold.split(X, Y):
        decisionTree = decisionTree.fit(X[train], Y[train])
        Y_pred = decisionTree.predict(X[test])
        cvscores.append(metrics.accuracy_score(Y[test], Y_pred) * 100)
        i += 1
    print("Mean(+/-Standard Deviation): %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Final test accuracy:", metrics.accuracy_score(Y[test], Y_pred) * 100)

if len(sys.argv) != 3:
    sys.exit("Must specify input and output files")
firstInputCSV = sys.argv[1]
secondInputCSV = sys.argv[2]

loadInput = np.loadtxt(firstInputCSV, delimiter=',', dtype='str', skiprows=1)
features = getFeatures(loadInput)

# metrics = np.zeros((10,2))
Y = features[10].values.reshape(-1, 1)
for i in range (10):
    X = features[i].values.reshape(-1, 1)
    print("Feature #%i" % (i+1))
    DecisionTree(X, Y, i)

# outputFilePath = "./" + str(secondInputCSV)
# with open(outputFilePath, 'w', newline='') as outputFile:
#     # writer = csv.writer(outputFile, delimiter=',')
#     df = pd.read_csv(firstInputCSV)
#     df['Hola'] = 'hola'
#     df.to_csv(outputFile, index=False)
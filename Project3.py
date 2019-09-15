import csv
import sys
import pandas as pd
import numpy as np
from FeatureExtraction import  getFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

seed = np.random.seed(1)

if len(sys.argv) != 3:
    sys.exit("Must specify input and output files")
firstInputCSV = sys.argv[1]
secondInputCSV = sys.argv[2]

loadInput = np.loadtxt(firstInputCSV, delimiter=',', dtype='str', skiprows=1)
features = getFeatures(loadInput)

decisionTree = DecisionTreeClassifier(criterion='entropy')
Y = features[10].values.reshape(-1, 1)
for i in range (10):
    X = features[i].values.reshape(-1, 1)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    # decisionTree = decisionTree.fit(X_train, y_train)
    # y_pred = decisionTree.predict(X_test)
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    scores = cross_val_score(estimator=decisionTree, X=X, y=Y, cv=5)
    print("Accuracy for feature %i: Mean %f, STD %f" % (i+1, scores.mean(), scores.std()))

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
# tree.plot_tree(clf.fit(iris.data, iris.target))

# outputFilePath = "./" + str(secondInputCSV)
# with open(outputFilePath, 'w', newline='') as outputFile:
#     # writer = csv.writer(outputFile, delimiter=',')
#     df = pd.read_csv(firstInputCSV)
#     df['Hola'] = 'hola'
#     df.to_csv(outputFile, index=False)
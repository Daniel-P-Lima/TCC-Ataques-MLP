import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from iTreeFile import exportClassifierTreeToFile, printClassifierTree

RANDOM_STATE = 42
USE_STD_DATA = True

if USE_STD_DATA:
    Xtrain = pd.read_csv("../Data/Xtrain.csv")
    Xtest = pd.read_csv("../Data/Xtest.csv")
    ytrain = pd.read_csv("../Data/ytrain.csv")
    ytest = pd.read_csv("../Data/ytest.csv")
else:
    X = pd.read_csv("../Data/iot23_combined.csv")

    y = X["label"]
    X = X.drop("label", axis=1)

    print(y.value_counts())

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

clf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=50, max_depth=None)

clf.fit(Xtrain, ytrain)

depths = [tree.get_depth() for tree in clf.estimators_]
print(depths)

exportClassifierTreeToFile(clf)

#print(clf.classes_)

# print(Xtest.iloc[[0]])
# print()
# y_pred = clf.predict(Xtest.iloc[[0]])

y_pred = clf.predict(Xtest)

# print(y_pred)

print(classification_report(ytest, y_pred))

Xtest.to_csv("../Data/Xtest.csv", index=False)

# cm = confusion_matrix(ytest, y_pred, labels=y.unique())

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
# disp.plot(values_format='d', cmap=None)
# plt.xticks(rotation=45)
# plt.title("Confusion Matrix — Random Forest")
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

X = pd.read_csv("../Data/iot23_combined.csv")

#row = X.loc[X.index[X["label"] == "C&C-Mirai"]]
#X = pd.concat([X, row], ignore_index=True)

X = X[X["label"] != "C&C-Mirai"]

y = X["label"]
X = X.drop("label", axis=1)

y = y.where(y == "Benign", "Attack")

print(y.value_counts())

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

clf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=50, max_depth=None)

clf.fit(Xtrain, ytrain)

y_pred = clf.predict(Xtest)

print(classification_report(ytest, y_pred))

cm = confusion_matrix(ytest, y_pred, labels=y.unique())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
disp.plot(values_format='d', cmap=None)
plt.xticks(rotation=45)
plt.title("Confusion Matrix — Random Forest")
plt.show()
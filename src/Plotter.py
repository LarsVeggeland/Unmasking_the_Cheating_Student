import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/curves/yesyes.csv")
labels = df["label"].to_list()
features = df.iloc[:,1:].to_numpy()

for i, curve in enumerate(features):
    color = "r" if labels[i] == True else "b"
    author = "same author curve" if labels[i] == True else "different author curve"
    plt.plot(curve, color=color)

plt.xlabel("Iterations")
plt.ylabel("Accuracy")

clf = SVC(kernel="linear")
scores = cross_val_score(clf, features, labels, cv=5)
print(scores)
print(f"Average model performance from 5-fold cross-validation: {round(100*scores.mean(), 2)} %")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.35, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"Confusion matrix for smaller sample:\ntn={tn}\nfp={fp}\nfn={fn}\ntp={tp}")

plt.show()
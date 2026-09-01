import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

os.makedirs("model", exist_ok=True)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

joblib.dump(clf, "model/model.joblib")
print(f"Model saved. Accuracy: {clf.score(X_test, y_test):.2f}")

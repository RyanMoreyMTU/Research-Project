from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

file_paths = [
    "CSVFeaturesChanged/eeg25_features_changed.csv",
    "CSVFeaturesChanged/eeg44_features_changed.csv",
    "CSVFeaturesChanged/eeg34_features_changed.csv",
    "CSVFeaturesChanged/eeg42_features_changed.csv"
]

dfs = [pd.read_csv(file) for file in file_paths]

df = pd.concat(dfs, ignore_index=True)

df.drop(['start', 'end'], axis=1, inplace=True)

X = df.drop('seizure_label', axis=1)
y = df['seizure_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svm_model = SVC(kernel='linear', random_state=0)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(np.sum(y_test)/len(y_test))
print(np.sum(y_pred)/len(y_pred))
print(y_pred)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_paths = [
    "~/ResearchProject/CSVFeaturesChanged/eeg25_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg44_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg34_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg42_features_changed.csv"
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

plt.figure(figsize=(10, 5))

plt.plot(y_test.values, label='y_test', color='blue', linestyle='-')

plt.plot(y_pred, label='y_pred', color='red', linestyle='-')

seizure_indices = np.where(y_test == 1)[0]
for idx in range(len(y_test)):
    if y_test.values[idx] == y_pred[idx]:
        plt.scatter(idx, y_test.values[idx], color='green', label='Agreement' if idx == seizure_indices[0] else '')

plt.title('Comparison between y_test and y_pred')
plt.xlabel('Index')
plt.ylabel('Annotation')
plt.legend()
plt.grid(True)
plt.show()


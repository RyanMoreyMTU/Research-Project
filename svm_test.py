from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

file_paths = {
    "~/ResearchProject/CSVFeaturesChanged/eeg25_features_changed.csv": "seizure",
    "~/ResearchProject/CSVFeaturesChanged/eeg44_features_changed.csv": "seizure",
    "~/ResearchProject/CSVFeaturesChanged/eeg34_features_changed.csv": "seizure",
    "~/ResearchProject/CSVFeaturesChanged/eeg42_features_changed.csv": "non-seizure",
    "~/ResearchProject/CSVFeaturesChanged/eeg73_features_changed.csv": "test"
}    
        
dfs = {path: pd.read_csv(path).drop(['start', 'end'], axis=1) for path in file_paths.keys()}
df = pd.concat(dfs.values(), ignore_index=True)

# Split up data frames into train and test
train_dfs = [dfs[path] for path, label in file_paths.items() if label != "test" and label != "non_seizure"]
test_df = dfs[next(path for path, label in file_paths.items() if label == "test")]

train_df = pd.concat(train_dfs, ignore_index=True)
print(train_df)
# For non-seizure data, take only 40% of the file's length
for path, label in file_paths.items():
    if label == "non_seizure":
        non_seizure_df = dfs[path]
        non_seizure_df = non_seizure_df[non_seizure_df['seizure_label'] == 0]
        num_samples_from_non_seizure = int(0.4 * len(non_seizure_df))
        non_seizure_sampled = non_seizure_df.sample(n=num_samples_from_non_seizure, random_state=0)
        train_df = pd.concat([train_df, non_seizure_sampled])

X_train = train_df.drop('seizure_label', axis=1)
y_train = train_df['seizure_label']

X_test = test_df.drop('seizure_label', axis=1)
y_test = test_df['seizure_label']

# Select only the features present in both datasets
# (they are the same anyways but doesn't run unless I do this for some reason)
common_features = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_features]
X_test = X_test[common_features]

# Normalize the features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(X_train_normalized, y_train)

y_pred = svm_model.predict(X_test_normalized)

# Start of coefficient/feature importance code
feature_names = X_train.columns

coefficients = svm_model.coef_[0]
coefficients_with_features = zip(coefficients, feature_names)
coefficients_with_features = sorted(coefficients_with_features, key=lambda x: abs(x[0]), reverse=True)

print("Feature Coefficients:")
for coef, feature_name in coefficients_with_features:
    if 'mean' in feature_name:
        print(f"{feature_name}: {coef}")

# Getting rid of isolated predictions
for i in range(len(y_pred) - 1):
    if y_pred[i] == 1 and y_pred[i + 1] == 0 and y_pred[i - 1] == 0:
        y_pred[i] = 0
    elif y_pred[i] == 1 and y_pred[i + 1] == 1 and y_pred[i - 1] == 0 and y_pred[i+2] == 0:
        y_pred[i] = 0

# Code for comparing actual answer to prediction results in a plot
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
plt.legend(loc='center right')
plt.grid(True)
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(np.sum(y_test)/len(y_test))
print(np.sum(y_pred)/len(y_pred))
print(y_pred)
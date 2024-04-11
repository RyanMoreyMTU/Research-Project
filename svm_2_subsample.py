from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_paths = [
    "~/ResearchProject/CSVFeaturesChanged/eeg25_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg44_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg34_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg42_features_changed.csv",
    "~/ResearchProject/CSVFeaturesChanged/eeg73_features_changed.csv"
]

dfs = [pd.read_csv(file).drop(['start', 'end'], axis=1) for file in file_paths]
df = pd.concat(dfs, ignore_index=True)

# Split up data frames into seizure and non seizure
seizure_df = pd.concat([df[df['seizure_label'] == 1] for df in dfs[:-1]])
non_seizure_df = pd.concat([df[df['seizure_label'] == 0] for df in dfs[:-1]])

# For each seizure dataframe, get an equal amount of seizure and non-seizure data
num_non_seizure_samples = len(seizure_df)
non_seizure_sampled = non_seizure_df.sample(n=num_non_seizure_samples, random_state=0)

# For non seizure data, take only 40% of the files length
fourth_file_non_seizure = dfs[-2][dfs[-2]['seizure_label'] == 0]
num_samples_from_fourth_file = int(0.4 * len(fourth_file_non_seizure))
fourth_file_non_seizure_sampled = fourth_file_non_seizure.sample(n=num_samples_from_fourth_file, random_state=0)

# Concat all the new dataframes
balanced_df = pd.concat([seizure_df, non_seizure_sampled, fourth_file_non_seizure_sampled], ignore_index=True)

X = balanced_df.drop('seizure_label', axis=1)
y = balanced_df['seizure_label']

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=0)

svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Start of coefficient/feature importance code
feature_names = X.columns

coefficients = svm_model.coef_[0]
coefficients_with_features = zip(coefficients, feature_names)
coefficients_with_features = sorted(coefficients_with_features, key=lambda x: abs(x[0]), reverse=True)

print("Feature Coefficients:")
for coef, feature_name in coefficients_with_features:
    print(f"{feature_name}: {coef}")

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

# Classification Report and Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(np.sum(y_test)/len(y_test))
print(np.sum(y_pred)/len(y_pred))
print(y_pred)

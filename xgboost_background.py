import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

directory = "CSVFeaturesChangedBackground/"
eeg_file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

# Define labels for each file
file_paths = {}
for file_path in eeg_file_paths:
    df = pd.read_csv(os.path.expanduser(file_path))
    if df['seizure_label'].isin([1]).all():
        file_paths[file_path] = "seizure"
    elif df['seizure_label'].isin([0]).all():
        file_paths[file_path] = "non_seizure"
    else:
        file_paths[file_path] = "test"


dfs = {path: pd.read_csv(path).drop(['start', 'end'], axis=1) for path in file_paths.keys()}
df = pd.concat(dfs.values(), ignore_index=True)

# Get the paths of seizure files
seizure_file_paths = [path for path, label in file_paths.items() if label == "seizure"]

# Randomly select 20 seizure files to exclude
random.seed(42)
seizure_files_to_exclude = random.sample(seizure_file_paths, 18)

# Split up data frames into train and test
train_dfs = [dfs[path] for path, label in file_paths.items() if label != "test" and path not in seizure_files_to_exclude]
test_df = dfs[next(path for path, label in file_paths.items() if label == "test")]

train_df = pd.concat(train_dfs, ignore_index=True)

X_train = train_df.drop('seizure_label', axis=1)
y_train = train_df['seizure_label']

X_test = test_df.drop('seizure_label', axis=1)
y_test = test_df['seizure_label']

# Select only the features present in both datasets
common_features = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_features]
X_test = X_test[common_features]

# Normalize the features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# XGBoost Model
xgb_model = XGBClassifier()
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [3],
    'min_child_weight': [1],
    'gamma': [0.5],
    'subsample': [0.8],
    'colsample_bytree': [1.0],
    'reg_alpha': [0.5],
    'reg_lambda': [0],
    'scale_pos_weight': [10],
    'base_score': [0.9],
    'booster': ['dart'],
    'importance_type': ['gain'],
    'tree_method': ['approx'],
    'validate_parameters': [True],
    'n_jobs': [-1]
}

grid = GridSearchCV(xgb_model, param_grid, refit=True, verbose=2)
grid.fit(X_train_normalized, y_train)
print(grid.best_params_)

# Prediction
y_pred = grid.predict(X_test_normalized)

# Classification Report and Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(np.sum(y_test)/len(y_test))
print(np.sum(y_pred)/len(y_pred))
print(y_pred)

# Print grid search results
print("Grid Search Results:")
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Score: {:.4f}".format(grid.best_score_))
print("Best Estimator: ", grid.best_estimator_)

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

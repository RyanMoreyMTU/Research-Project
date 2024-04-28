import os
import pandas as pd

def process_file(file_path):
    df = pd.read_csv(file_path)
    if 1 in df['seizure_label'].values:
        df = df[df['seizure_label'] != 1]
        df['seizure_label'] = 1
    
    output_file_path = f"CSVFeaturesChangedBackground/{os.path.basename(file_path)}"
    df.to_csv(output_file_path, index=False)
    print(f"Processed file saved to {output_file_path}")

# list to add more processed feature files to if needed
feature_files = ['CSVFeaturesChanged/eeg25_features_changed.csv', 
                 'CSVFeaturesChanged/eeg44_features_changed.csv', 
                 'CSVFeaturesChanged/eeg72_features_changed.csv',
                 'CSVFeaturesChanged/eeg34_features_changed.csv',
                 'CSVFeaturesChanged/eeg42_features_changed.csv', 
                 'CSVFeaturesChanged/eeg58_features_changed.csv', 
                 'CSVFeaturesChanged/eeg3_features_changed.csv',
                 'CSVFeaturesChanged/eeg73_features_changed.csv',
                 'CSVFeaturesChanged/eeg56_features_changed.csv',
                 'CSVFeaturesChanged/eeg1_features_changed.csv',
                 'CSVFeaturesChanged/eeg4_features_changed.csv',
                 'CSVFeaturesChanged/eeg7_features_changed.csv']
# main loop
for file_path in feature_files:
    process_file(file_path)

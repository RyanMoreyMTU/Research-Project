import os
import pandas as pd

def process_file(file_path):
    df = pd.read_csv(file_path)
    counter = 0
    if 1 in df['seizure_label'].values:
        df = df[df['seizure_label'] != 1]
        df['seizure_label'] = 1
        counter += 1
    output_file_path = f"CSVFeaturesChangedBackground/{os.path.basename(file_path)}"
    df.to_csv(output_file_path, index=False)
    print(f"Processed file saved to {output_file_path}")
    return counter

# Directory containing processed feature files
directory = 'CSVFeaturesChanged/'

# Get all files in the directory
feature_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

# main loop
total_counter = 0
for file_path in feature_files:
    total_counter += process_file(file_path)

print(f"Total files processed: {len(feature_files)}")
print(f"Total files modified: {total_counter}")

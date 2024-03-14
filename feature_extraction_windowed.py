import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_features(channel_data):
    features = {}
    
    # curve length
    features['curve_length'] = np.sum(np.abs(np.diff(channel_data)))

    # root mean squared amplitude
    features['rms_amplitude'] = np.sqrt(np.mean(np.square(channel_data)))

    # zero crossings
    features['zero_crossings'] = np.sum(np.diff(np.sign(channel_data)) != 0)

    # skewness
    features['skewness'] = skew(channel_data)
    
    # kurtosis
    features['kurtosis'] = kurtosis(channel_data)

    # variance
    features['variance'] = np.var(channel_data)
    
    f, Pxx = welch(channel_data, fs=32, nperseg=len(channel_data))
    
    # total power (0–12 Hz)
    total_power_index = np.where((f >= 0) & (f <= 12))[0]
    total_power = np.trapz(Pxx[total_power_index], f[total_power_index])
    features['total_power'] = total_power
    
    # peak frequency of spectrum
    peak_frequency_index = np.argmax(Pxx)
    peak_frequency = f[peak_frequency_index]
    features['peak_frequency'] = peak_frequency

    return features

def process_row(row):
    # if row['mean_peak_frequency'] == 16:
    if row['mean_total_power'] < 1e-20:
        row.loc[row.index.difference(['start', 'end'])] = 0
    return row

# list to add more eeg file to if needed
eeg_file_paths = ['eeg25.csv', 
                  'eeg44.csv', 
                  'eeg72.csv',
                  'eeg34.csv',
                  'eeg42.csv', 
                  'eeg58.csv', 
                  'eeg3.csv',
                  'eeg73.csv',
                  'eeg56.csv']

# main loop
for file_path in eeg_file_paths:
    # readingin csv file
    df = pd.read_csv(file_path)
    
    # converting to numeric in case the scientific notation causes issues (e-10 etc.)
    eeg_columns = df.columns[1:-1]
    df[eeg_columns] = df[eeg_columns].apply(pd.to_numeric, errors='coerce')
    
    # initializing the seizure label to 0
    df['seizure_label'] = 0
    
    # creating the window size and making it modular incase I want to change it
    window_size = 60 * 32
    overlap_percentage = 50
    step_size = int(window_size * (1-overlap_percentage / 100))
    
    # the loop for the windowing
    feature_list = []
    for start in range(0, len(df), step_size):
        end = start + window_size
        
        if end > len(df):
            end = len(df)

        if 1 in df['Seizure'].iloc[start:end].values:
            df.loc[start:end, 'seizure_label'] = 1

        window_df = df.iloc[start:end, :]
        print(f"Processing window from {start} to {end} in file {file_path}")

        window_features = {'start': start, 'end': end, 'seizure_label': df['seizure_label'].iloc[start]}
        
        # looping through channels and extracting features
        for channel in eeg_columns:
            channel_data = window_df[channel].values
            channel_features = extract_features(channel_data)
            
            # looping through the channel features and creating a column for them
            for feature_name, feature_value in channel_features.items():
                column_name = f'{channel} {feature_name}'
                window_features[column_name] = feature_value

        feature_list.append(window_features)

    windowed_feature_df = pd.DataFrame(feature_list)
    
    feature_names = ['curve_length', 
                     'rms_amplitude', 
                     'zero_crossings', 
                     'skewness', 
                     'kurtosis', 
                     'variance', 
                     'total_power', 
                     'peak_frequency']
    
    # getting the mean of each feature for each window
    for feature_name in feature_names:
        channel_features = [col for col in windowed_feature_df.columns if f'{feature_name}' in col]
        windowed_feature_df[f'mean_{feature_name}'] = windowed_feature_df[channel_features].mean(axis=1)

   # Create a copy of the DataFrame before changes
    original_windowed_feature_df = windowed_feature_df.copy()

    # Save the original features to a CSV file
    output_file_path = f"{file_path.replace('.csv', '_features.csv')}"
    original_windowed_feature_df.to_csv(output_file_path, index=False)
    print(f"Original features saved to {output_file_path}")

    # Apply the process_row function to each row
    windowed_feature_df = windowed_feature_df.apply(process_row, axis=1)

    # Save the modified features to a new CSV file
    output_changed_file_path = f"{file_path.replace('.csv', '_features_changed.csv')}"
    windowed_feature_df.to_csv(output_changed_file_path, index=False)
    print(f"Processed features with changed values saved to {output_changed_file_path}")
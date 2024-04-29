import mne
import pandas as pd
import os

def add_filters(raw):
    raw.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=dict(ftype='butter', order=6),
               phase='zero', picks='all')

    raw.notch_filter(freqs=50, picks='all', notch_widths=4.0 / 256.0)

def standardize_channel_names(raw):
    raw.rename_channels(lambda x: x.lower())

def identify_channels_to_drop(raw):
    channels_to_drop = [ch_name for ch_name in raw.ch_names if 'Cz' in ch_name or 'EKG' in ch_name or 'Effort' in ch_name]
    return channels_to_drop

def clean(input_file, output_file, annotation_file, eeg_column):
    raw = mne.io.read_raw_edf(input_file, preload=True)
    add_filters(raw)
    channels_to_drop = identify_channels_to_drop(raw)
    raw.drop_channels(channels_to_drop)

    standardize_channel_names(raw)

    annotations = pd.read_csv(annotation_file)
    onsets = annotations[annotations[eeg_column] == 1].index

    raw.set_annotations(mne.Annotations(onset=onsets, duration=1, description='Seizure'))

    # new code for adding antialiasing 
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw = raw.resample(sfreq=32)
    raw.pick(picks=eeg_picks)

    times = raw.times
    df = pd.DataFrame(data=raw.get_data().T, columns=raw.ch_names)
    df['time'] = times
    df.set_index('time', inplace=True)

    df.index = pd.to_datetime(df.index, unit='s')
    df_resampled = df.resample('0.03125S').asfreq().reset_index()

    df_resampled['Seizure'] = 0

    onsets_indices = df_resampled['time'].searchsorted(pd.to_datetime(onsets, unit='s'))

    for idx in onsets_indices:
        df_resampled.loc[idx:idx + 31, 'Seizure'] = 1
    df_resampled.to_csv(output_file, index=False)

# Directory containing EDF files
edf_directory = 'EDFFiles/'

# Get a list of all .edf files in the directory
edf_files = [f for f in os.listdir(edf_directory) if f.endswith('.edf')]

# Generate eeg_files_info list
eeg_files_info = [{'input_file': os.path.join(edf_directory, file),
                   'output_file': 'CSVRaw/' + file.replace('.edf', '.csv'),
                   'eeg_column': file.replace('eeg', '').replace('.edf', '')}
                  for file in edf_files]

# Iterate through eeg_files_info and call clean function for each file
for info in eeg_files_info:
    clean(info['input_file'], info['output_file'], 'Annotations/annotationC.csv', info['eeg_column'])
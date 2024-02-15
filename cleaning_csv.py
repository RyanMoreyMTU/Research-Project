import mne
import pandas as pd

def add_filters(raw):
    raw.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=dict(ftype='butter', order=6),
               phase='zero', picks='all')

    raw.notch_filter(freqs=50, picks='all', notch_widths=4.0 / 256.0)
    
# using .lower() to standardize the channels
def standardize_channel_names(raw):
    raw.rename_channels(lambda x: x.lower())

# instead of two separate function, just checking if a unique common phrase is
# found in each channel and dropping it if so.
def identify_channels_to_drop(raw):
    channels_to_drop = [ch_name for ch_name in raw.ch_names if 'Cz' in ch_name or 'EKG' in ch_name or 'Effort' in ch_name]
    return channels_to_drop

# gets raw data, drops channels, standardizes, annotates, adds filters, adds times,
# and converts to csv
def clean(input_file, output_file, annotation_file, eeg_column):
    raw = mne.io.read_raw_edf(input_file, preload=True)
    add_filters(raw)
    channels_to_drop = identify_channels_to_drop(raw)
    raw.drop_channels(channels_to_drop)
    
    standardize_channel_names(raw)
    
    annotations = pd.read_csv(annotation_file)
    onsets = annotations[annotations[eeg_column] == 1].index
    
    raw.set_annotations(mne.Annotations(onset=onsets, duration=1, description='Seizure'))

    times = raw.times
    df = pd.DataFrame(data=raw.get_data().T, columns=raw.ch_names)
    df['time'] = times
    df.set_index('time', inplace=True)

    df.index = pd.to_datetime(df.index, unit='s')
    df_resampled = df.resample('1S').asfreq().reset_index()

    df_resampled['Seizure'] = df_resampled['time'].isin(pd.to_datetime(onsets, unit='s')).astype(int)

    df_resampled.to_csv(output_file, index=False)


eeg_files_info = [
    {'input_file': 'eeg25.edf', 'output_file': 'eeg25_csv.csv', 'eeg_column': '25'},
    {'input_file': 'eeg44.edf', 'output_file': 'eeg44_csv.csv', 'eeg_column': '44'},
    {'input_file': 'eeg34.edf', 'output_file': 'eeg34_csv.csv', 'eeg_column': '34'},
    {'input_file': 'eeg42.edf', 'output_file': 'eeg42_csv.csv', 'eeg_column': '42'},
    {'input_file': 'eeg58.edf', 'output_file': 'eeg58_csv.csv', 'eeg_column': '58'},
    {'input_file': 'eeg72.edf', 'output_file': 'eeg72_csv.csv', 'eeg_column': '72'},
    {'input_file': 'eeg3.edf', 'output_file': 'eeg3_csv.csv', 'eeg_column': '3'},
    {'input_file': 'eeg73.edf', 'output_file': 'eeg73_csv.csv', 'eeg_column': '73'},
    {'input_file': 'eeg56.edf', 'output_file': 'eeg56_csv.csv', 'eeg_column': '56'}
]

for info in eeg_files_info:
    clean(info['input_file'], info['output_file'], 'merged_annotation.csv', info['eeg_column'])


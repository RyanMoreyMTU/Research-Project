import mne
import pandas as pd

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

eeg_files_info = [
    {'input_file': 'EDFFiles/eeg25.edf', 'output_file': 'CSVRaw/eeg25.csv', 'eeg_column': '25'},
    {'input_file': 'EDFFiles/eeg44.edf', 'output_file': 'CSVRaw/eeg44.csv', 'eeg_column': '44'},
    {'input_file': 'EDFFiles/eeg34.edf', 'output_file': 'CSVRaw/eeg34.csv', 'eeg_column': '34'},
    {'input_file': 'EDFFiles/eeg42.edf', 'output_file': 'CSVRaw/eeg42.csv', 'eeg_column': '42'},
    {'input_file': 'EDFFiles/eeg58.edf', 'output_file': 'CSVRaw/eeg58.csv', 'eeg_column': '58'},
    {'input_file': 'EDFFiles/eeg72.edf', 'output_file': 'CSVRaw/eeg72.csv', 'eeg_column': '72'},
    {'input_file': 'EDFFiles/eeg3.edf', 'output_file': 'CSVRaw/eeg3.csv', 'eeg_column': '3'},
    {'input_file': 'EDFFiles/eeg73.edf', 'output_file': 'CSVRaw/eeg73.csv', 'eeg_column': '73'},
    {'input_file': 'EDFFiles/eeg56.edf', 'output_file': 'CSVRaw/eeg56.csv', 'eeg_column': '56'}
]

for info in eeg_files_info:
    clean(info['input_file'], info['output_file'], 'Annotations/annotationA.csv', info['eeg_column'])

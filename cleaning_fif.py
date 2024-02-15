import mne
import pandas as pd
# separate functions because they dont all share the same channel names

def add_filters(raw):
    raw.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=dict(ftype='butter', order=6),
               phase='zero', picks='all')

    raw.notch_filter(freqs=50, picks='all', notch_widths=4.0 / 256.0)
    
def clean_eeg1_eeg2(input_file, output_file, annotation_file, eeg_column):
    # loading edf
    raw = mne.io.read_raw_edf(input_file, preload=True)
    
    # getting rid of the channels that dont seem to make a big impact
    channels_to_drop = ['EEG Cz-REF', 'ECG EKG-REF', 'Resp Effort-REF']  
    raw.drop_channels(channels_to_drop)
    
    # loading csv
    annotations = pd.read_csv(annotation_file)
    onsets = annotations[annotations[eeg_column] == 1].index
    
    # annotating the data
    raw.set_annotations(mne.Annotations(onset=onsets, duration=1, description='Seizure'))
    add_filters(raw)
    raw.save(output_file, overwrite=True)


def clean_eeg3_eeg6(input_file, output_file, annotation_file, eeg_column):
    raw = mne.io.read_raw_edf(input_file, preload=True)

    channels_to_drop = ['EEG Cz-Ref', 'ECG EKG', 'Resp Effort']
    raw.drop_channels(channels_to_drop)
    
    annotations = pd.read_csv(annotation_file)
    onsets = annotations[annotations[eeg_column] == 1].index
    
    raw.set_annotations(mne.Annotations(onset=onsets, duration=1, description='Seizure'))
    add_filters(raw)
    raw.save(output_file, overwrite=True)


eeg1_annotation_file = 'eeg1_annotations.csv'
eeg2_annotation_file = 'eeg2_annotations.csv'
eeg3_annotation_file = 'eeg3_annotations.csv'
eeg4_annotation_file = 'eeg4_annotations.csv'
eeg5_annotation_file = 'eeg5_annotations.csv'
eeg6_annotation_file = 'eeg6_annotations.csv'

eeg1_input_file = 'eeg1.edf'
eeg1_output_file = 'eeg1_raw.fif'
clean_eeg1_eeg2(eeg1_input_file, eeg1_output_file, eeg1_annotation_file, 'eeg1')

eeg2_input_file = 'eeg2.edf'
eeg2_output_file = 'eeg2_raw.fif'
clean_eeg1_eeg2(eeg2_input_file, eeg2_output_file, eeg2_annotation_file, 'eeg2')

eeg3_input_file = 'eeg3.edf'
eeg3_output_file = 'eeg3_raw.fif'
clean_eeg3_eeg6(eeg3_input_file, eeg3_output_file, eeg3_annotation_file, 'eeg3')

eeg4_input_file = 'eeg4.edf'
eeg4_output_file = 'eeg4_raw.fif'
clean_eeg3_eeg6(eeg4_input_file, eeg4_output_file, eeg4_annotation_file, 'eeg4')

eeg5_input_file = 'eeg5.edf'
eeg5_output_file = 'eeg5_raw.fif'
clean_eeg3_eeg6(eeg5_input_file, eeg5_output_file, eeg5_annotation_file, 'eeg5')

eeg6_input_file = 'eeg6.edf'
eeg6_output_file = 'eeg6_raw.fif'
clean_eeg3_eeg6(eeg6_input_file, eeg6_output_file, eeg6_annotation_file, 'eeg6')

import mne

# Function to clean EEG data for eeg1 and eeg2
def clean_eeg1_eeg2(input_file, output_file):
    # Loading EDF file
    raw = mne.io.read_raw_edf(input_file, preload=True)
    
    # Dropping unnecessary channels
    channels_to_drop = ['EEG Cz-REF', 'ECG EKG-REF', 'Resp Effort-REF']  
    raw.drop_channels(channels_to_drop)
    
    # Saving with MNE's .save function
    raw.save(output_file, overwrite=True)


def clean_eeg3_eeg6(input_file, output_file):
    raw = mne.io.read_raw_edf(input_file, preload=True)

    channels_to_drop = ['EEG Cz-Ref', 'ECG EKG', 'Resp Effort']
    raw.drop_channels(channels_to_drop)

    raw.save(output_file, overwrite=True)


eeg1_input_file = 'eeg1.edf'
eeg1_output_file = 'eeg1_cleaned.fif'
clean_eeg1_eeg2(eeg1_input_file, eeg1_output_file)

eeg2_input_file = 'eeg2.edf'
eeg2_output_file = 'eeg2_cleaned.fif'
clean_eeg1_eeg2(eeg2_input_file, eeg2_output_file)

eeg3_input_file = 'eeg3.edf'
eeg3_output_file = 'eeg3_cleaned.fif'
clean_eeg3_eeg6(eeg3_input_file, eeg3_output_file)

eeg4_input_file = 'eeg4.edf'
eeg4_output_file = 'eeg4_cleaned.fif'
clean_eeg3_eeg6(eeg4_input_file, eeg4_output_file)

eeg5_input_file = 'eeg5.edf'
eeg4_output_file = 'eeg5_cleaned.fif'
clean_eeg3_eeg6(eeg4_input_file, eeg4_output_file)

eeg6_input_file = 'eeg6.edf'
eeg4_output_file = 'eeg6_cleaned.fif'
clean_eeg3_eeg6(eeg4_input_file, eeg4_output_file)

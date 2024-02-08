import mne

def verify_fif_file(file_path):
    raw = mne.io.read_raw_fif("eeg4_cleaned.fif", preload=True)

    print(raw.info)

if __name__ == "__main__":
    file_path = 'your_file.fif'
    
    verify_fif_file(file_path)

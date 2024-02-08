import mne
import matplotlib.pyplot as plt

def plot_annotations_static(input_file):
    raw = mne.io.read_raw_fif(input_file, preload=True)
  
    annotations = raw.annotations
    onset = annotations.onset
    duration = annotations.duration
    fig, ax = plt.subplots()

    raw.plot(duration=1, n_channels=30, scalings='auto', show=False)

    for onset_time, duration_time in zip(onset, duration):
        ax.axvline(onset_time, color='r', linestyle='--', alpha=0.7, linewidth=2, ymin=0, ymax=0.8)

    plt.show()

plot_annotations_static('eeg1_raw.fif')

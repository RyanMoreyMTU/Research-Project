import pandas as pd
import matplotlib.pyplot as plt

def plot_eeg_data(df, channels_to_plot, title):
    df_subset = df.head(1920)

    fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(10, 6), sharex=True)

    for i, channel in enumerate(channels_to_plot):
        axs[i].plot(df_subset[channel], label=channel)
        axs[i].set_ylabel('EEG Amplitude')
        axs[i].legend()

    axs[-1].set_xlabel('Sample Index')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


df = pd.read_csv('eeg25_mne.csv')

channels_to_plot = ['eeg fp1-ref', 'eeg fp2-ref', 'eeg f3-ref', 'eeg f4-ref']

plot_eeg_data(df, channels_to_plot, 'EEG Data')
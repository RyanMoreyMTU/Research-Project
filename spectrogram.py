import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_eeg_data(df, channels_to_plot, title):
    df_subset = df.head(300)

    fig, axs = plt.subplots(len(channels_to_plot), 2, figsize=(14, 6), sharex=True)

    for i, channel in enumerate(channels_to_plot):
        axs[i, 0].plot(df_subset[channel], label=channel)
        axs[i, 0].set_ylabel('Amplitude')
        axs[i, 0].legend()

        f, Pxx = welch(df_subset[channel], fs=1.0, nperseg=64)
        axs[i, 1].imshow([Pxx], aspect='auto', cmap='viridis', extent=[0, len(df_subset), f[0], f[-1]])
        axs[i, 1].set_ylabel('Frequency (Hz)')

    axs[0, 0].set_title('Time-Domain Signal')
    axs[0, 1].set_title('Welch Spectrogram')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
both_filters = 'eeg25_both_filters.csv'
butterworth = 'eeg25_butterworth.csv'
no_filters = 'eeg25_no_filters.csv'
notch = 'eeg25_notch.csv'

df_both = pd.read_csv(both_filters)
df_butterworth = pd.read_csv(butterworth)
df_none = pd.read_csv(no_filters)
df_notch = pd.read_csv(notch)

channels_to_plot = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG F3-REF', 'EEG F4-REF']

plot_eeg_data(df_none, channels_to_plot, 'No Filters')
plot_eeg_data(df_notch, channels_to_plot, 'Notch Filter')
plot_eeg_data(df_butterworth, channels_to_plot, 'Butterworth Filter')
plot_eeg_data(df_both, channels_to_plot, 'Both Filters')

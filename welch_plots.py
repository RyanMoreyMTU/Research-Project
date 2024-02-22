import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_eeg_data(df, channels_to_plot, title, fs=256):
    df['time'] = pd.to_datetime(df['time'])
    df_subset = df#df.head(300)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    #fs = 256.0
    for i, channel in enumerate(channels_to_plot):
        f, Pxx = welch(df_subset[channel], fs, nperseg=256)
        axs[0].semilogy(f, Pxx, label=f'{channel} - PSD')

    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('PSD (V^2/Hz)')
    axs[0].legend()

    for i, channel in enumerate(channels_to_plot):
        f, Pxx = welch(df_subset[channel], fs, nperseg=256) 
        axs[1].plot(f, Pxx, label=f'{channel} - Power Spectrum')

    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Power (V^2)')
    axs[1].legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

both_filters = 'high_pass_notch_resample.csv'
butterworth = 'high_pass.csv'
no_filters = 'raw.csv'
notch = 'high_pass_notch.csv'

df_both = pd.read_csv(both_filters)
df_butterworth = pd.read_csv(butterworth)
df_none = pd.read_csv(no_filters)
df_notch = pd.read_csv(notch)

channels_to_plot = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG F3-REF', 'EEG F4-REF']

plot_eeg_data(df_none, channels_to_plot, 'No Filters')
plot_eeg_data(df_butterworth, channels_to_plot, 'Butterworth Filter')
plot_eeg_data(df_notch, channels_to_plot, 'Notch Filter')
plot_eeg_data(df_both, channels_to_plot, 'Both Filters (down sample)', fs=32)

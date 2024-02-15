import pandas as pd
import matplotlib.pyplot as plt
    
def plot_eeg_data(df, channels_to_plot, title):
    df_subset = df.head(300)

    fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(10, 6), sharex=True)

    for i, channel in enumerate(channels_to_plot):
        axs[i].plot(df_subset[channel], label=channel)
        axs[i].set_ylabel('amplitude')
        axs[i].legend()

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

# Notch is used to get rid of specific frequencies, in this case, power line interference.
# It can affect some channels more than others.

# High pass butterworth weakens signals that go above a certain threshold.

# raw.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=dict(ftype='butter', order=6),
#           phase='zero', picks='all')

# raw.notch_filter(freqs=50, picks='all', notch_widths=4.0 / 256.0)

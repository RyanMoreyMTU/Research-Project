import mne
import pandas as pd
import numpy as np


def add_filters(raw):
    raw.filter(
        l_freq=0.5,
        h_freq=None,
        method="iir",
        iir_params=dict(ftype="butter", order=6),
        phase="zero",
        picks="all",
    )

    raw.notch_filter(freqs=50, picks="all", notch_widths=4.0 / 256.0)


def standardize_channel_names(raw):
    raw.rename_channels(lambda x: x.lower())


def identify_channels_to_drop(raw):
    channels_to_drop = [
        ch_name
        for ch_name in raw.ch_names
        if "Cz" in ch_name or "EKG" in ch_name or "Effort" in ch_name
    ]
    return channels_to_drop


def clean(input_file, output_file, annotation_file, eeg_column):
    raw = mne.io.read_raw_edf(input_file, preload=True)
    add_filters(raw)
    channels_to_drop = identify_channels_to_drop(raw)
    raw.drop_channels(channels_to_drop)

    standardize_channel_names(raw)

    annotations = pd.read_csv(annotation_file)

    anno_col = annotations[eeg_column].to_numpy()  # Extract correct anno column
    anno_col = anno_col[
        ~np.isnan(anno_col)
    ]  # remove NaN - some rows in anno file are blank due to differing sizes
    anno = np.repeat(anno_col, 32).astype(int)  # Resample to 32 Hz

    # new code for adding antialiasing
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw = raw.resample(sfreq=32, npad="auto", window="boxcar", n_jobs=1)
    raw.pick(picks=eeg_picks)

    times = raw.times
    df = pd.DataFrame(data=raw.get_data().T, columns=raw.ch_names)
    df["time"] = times
    df.set_index("time", inplace=True)

    # Save cleaned data and annotations to different files
    df.to_csv(output_file, index=False)
    np.savetxt(output_file.split(".")[0] + "_anno.csv", anno, delimiter=",", fmt="%i")


eeg_files_info = [
    {"input_file": "eeg25.edf", "output_file": "eeg25.csv", "eeg_column": "25"},
    {"input_file": "eeg44.edf", "output_file": "eeg44.csv", "eeg_column": "44"},
    {"input_file": "eeg34.edf", "output_file": "eeg34.csv", "eeg_column": "34"},
    {"input_file": "eeg42.edf", "output_file": "eeg42.csv", "eeg_column": "42"},
    {"input_file": "eeg58.edf", "output_file": "eeg58.csv", "eeg_column": "58"},
    {"input_file": "eeg72.edf", "output_file": "eeg72.csv", "eeg_column": "72"},
    {"input_file": "eeg3.edf", "output_file": "eeg3.csv", "eeg_column": "3"},
    {"input_file": "eeg73.edf", "output_file": "eeg73.csv", "eeg_column": "73"},
    {"input_file": "eeg56.edf", "output_file": "eeg56.csv", "eeg_column": "56"},
]

for info in eeg_files_info:
    clean(
        info["input_file"], info["output_file"], "annotationA.csv", info["eeg_column"]
    )

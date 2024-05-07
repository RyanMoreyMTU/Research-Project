# Classifying Seizure Neonates Based Off Their Background EEG Signals

This projects goal is to classify if an neonate (newborn infant) is prone to seizures based off the background data of its EEG signals.

All the data used in this project was acquired from: https://zenodo.org/records/2547147
It's a data set with 79 EDF files of EEG data. For the sake of accuracy, I only used the files that all 3 expert annotators agreed upon as shown in the clinical_infomration file.

The project directory should have sub-directories for each type of data file:
Annotations/ - Directory to store the annotation files (A, B, C).
EDFFiles/ - Directory to store the EDF files.
CSVRaw/ - Directory to output the files generated from the cleaning_done.py file.
CSVFeatures/ - Directory to store the unaltered features from the feature_extraction_windowed.py file.
CSVFeaturesChanged/ - Directory to store the altered features from the feature_extraction_windowed.py file.
CSVFeaturesChangedBackground/ - Directory to store the files generated from the feature_extraction_background.py file.

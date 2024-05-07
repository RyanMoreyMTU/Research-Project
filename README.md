# Classifying Seizure Neonates Based Off Their Background EEG Signals

This projects goal is to classify if an neonate (newborn infant) is prone to seizures based off the background data of its EEG signals.

All the data used in this project was acquired from: https://zenodo.org/records/2547147
It's a data set with 79 EDF files of EEG data. For the sake of accuracy, I only used the files that all 3 expert annotators agreed upon as shown in the clinical_infomration file.

## Directory Structure
The project directory should have sub-directories for each type of data file: <br>
Annotations/ - Directory to store the annotation files (A, B, C). <br>
EDFFiles/ - Directory to store the EDF files. <br>
CSVRaw/ - Directory to output the files generated from the cleaning_done.py file. <br>
CSVFeatures/ - Directory to store the unaltered features from the feature_extraction_windowed.py file. <br>
CSVFeaturesChanged/ - Directory to store the altered features from the feature_extraction_windowed.py file. <br>
CSVFeaturesChangedBackground/ - Directory to store the files generated from the feature_extraction_background.py file. <br>

Optional: <br>
CSVFeaturesBackground/ - Directory to store the files generated from the feature_extraction_background.py file if you wish to use the unaltered data. <br>

## Instructions
Once the directory structure is properly setups, you can clone the repo.

### Cleaning File
The cleaning_done.py file is the first file that should be run. <br>
This file will read in the EDF files, clean them, and output the new data to a CSV file. <br>

The cleaning includes: <br>
Adding filters to the channel data. <br>
Standardizing the channel names. <br>
Dropping unneeded channels. <br>
Annotating the data. <br>
Resampling the data to 32hz (from 256hz). <br>
Adding a time column. <br>

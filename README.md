# Classifying Seizure Neonates Based Off Their Background EEG Signals

This projects goal is to classify if a neonate (newborn infant) is prone to seizures based off the background data of its EEG signals.

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
The **cleaning_done.py** file is the first file that should be run. <br>
This file will read in the EDF files, clean them, and output the new data to a CSV file. <br>

The cleaning includes: <br>
Adding filters to the channel data. <br>
Standardizing the channel names. <br>
Dropping unneeded channels. <br>
Annotating the data. <br>
Resampling the data to 32hz (from 256hz). <br>
Adding a time column. <br>

### Feature Extraction
The **feature_extraction_windowed.py** file is the next file that should be run. <br>
It will extract 12 features, the mean, min, and max of each row, and then output to 2 directories. <br>
The first directory, CSVFeatures, is the unaltered data.  <br>
The second directory, CSVFeaturesChanged, is the altered data. This data is altered because it accounts for unforseen flatlines in the data that may be harmful. The flatlines are identified via the mean_total_power column having a value much lower than the other rows. <br>

Features Extracted:
Curve Length <br>
Spectral Entropy <br>
RMS (Root Mean Squared) <br>
Zero Crossings <br>
Skewness <br>
Kurtosis <br>
Variance <br>
Total Power <br>
Peak Frequency of Spectrum <br>
Hjorth Paramters (activity, mobility, complexity) <br>

The feature extraction also uses windowing at 60 seconds per window with a 50% overlap. This means that the first wil be from 0-60 and the second window will be from 30-90 and so on.  <br>

The **feature_extraction_background.py** file will convert the files in CSVFeaturesChanged so that the seizure files will have no seizures and the whole file will be converted to 1s to indicate that its a background EEG file. You can change the input and output directories if you want to use the unaltered data in the CSVFeatures directory instead.

### Models
At this point, you should have all the data you want and you can run any of the model files.  <br>

I didn't work on the normal seizure detection model as much because optimizing them wasn't my goal for this project. I wanted to get used to workin with EEG signal with ML models and normal seizure detection models have a lot more documentation online than background EEG models. When I was comfortable enough, I moved onto my main focus which was background EEG models.

Model Files As Of 07/05/2024:
svm_2.py - First SVM model, very basic, normal seizure detection, only uses 5 files <br>
svm_2_subsample.py - Second SVM model, balances the seizure file data and subsamples the non seizure data, normal seizure detection, only uses 5 files <br>
svm_test.py - Third SVM model, only subsamples the non seizure data, normal seizure detection, only uses 5 files <br>
svm_background - SVM model for background EEG, takes in all the files, gridsearch was utilised  <br>
xgboost_background - XGboost model for background EEG, takes in all the files, gridsearch was utilised <br>

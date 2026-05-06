- Jakes Model Trainer & Visualizer -
  Version - 3.2

Updated Version3.2 Patches:
 - Fixed chart plotting issues in gui_logic
 - Fixed variable initialization issues in gui_logic and train_sgd_inchunks
 - Fixed display issues in startup nodule (modelmakermain.py)
 - Updated display labels in ModelMakeGuiV3.ui

Description:
This program generates a window via PyQt6 GUI to make visualizing the impact of shifting model weight more user friendly.
Provided on github is a preprocessing script for cicids2017_cleaned.csv data set; The script is designed to be ran from the same directory as the dataset
cicids2017_cleaned.csv dataset must be downloaded to the same directory prior to preprocessing and preprocessing.py must before running the GUI via modelmakermain.py
Data set can be downloaded here --> https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed 

Required Python Imports:
 - numpy
 - pandas
 - PyQt6
 - joblib
 - matplotlib
 - os
 - sklearn
 - sys
 - time

Steps:
1. Download cicids2017_cleaned.csv
2. Run preprocess.py from the same directory where cicids2017 is located
3. Install Imports & Dependencies
4. Run Startup Module (modelmakermain.py) in the same directory where cicids was preprocessed




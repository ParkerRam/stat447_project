# STAT 447 Project
 
Download the data [here](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_dataset_Summary.csv) and setup directory like the following.

The script transforming the data for the models takes quite long so pkl files of the transformed data can be downloaded [here](https://drive.google.com/drive/folders/1E-YuvdQqDUIankhK5OkuN6TPJHYNBEY-?usp=sharing) and placed in the directory as below.
```
stat447_project
│   README.md
│   requirements.txt
|   *.py
│
└───data
    │   Chest_xray_Corona_dataset_Summary.csv
    │   Chest_xray_Corona_Metadata.csv
    |   *.pkl
    │
    └───train
    │        ...
    │
    └───test
             ...
```
  
  ### Running code and performing comparison
 1. Set up virtual environment and activate environment (see instructions below)
 2. Run exploratory analysis and data processing with 
 
     ```
      python3 exploratory_analysis.py
      ```
     (Note: can skip to step 4 by downloading the files linked above since this step takes a long time.)
     - Reads, cleans and processes raw data set from Kaggle project
     - Splits data into training and test sets
     - Performs image augmentation on training set
     - Performs feature engineering (Note: this takes the longest to run)
     - Creates histogram and box-plots of feature distributions
     - Saves data .pkl files to be used for modelling
 3. Fit and predict using statistical methods with
     ```
      python3 modelling.py
      ```
      - Using transformed data, performs multinomial classification with 4 classes: "Bacteria", "COVID-19", "Other Virus", "Healthy"
      - Each of the methods below result in 2 models using balanced class weights. 1 model uses all features, while the other uses a subset.
        - Logistic Regression
        - Random Forest
        - Ada Boost
      - Each methods uses hyperparameter tuning using grid search 4-fold cross-validation (Note: this part takes the longest to run)
      - Various out-of-sample performance measures are calculated to be used for comparison
 4. Fit and predict using Convolutional Neural Networks (CNN) with
       ```
       python3 cnn.py
       ```
       - Splits data to training, validation test set, and performs image augmentation
       - trains and predicts a CNN model
       - calculates out-of-sample performance measures for comparison
 
 
### Setting up virtual environment
1. You can choose this option if your have python version 3.7.x or 3.8.x installed on your system. To find out, run
     ```
     python3 --version
     ```
 
2. Install `Virtualenv` with
     ```
     python3 -m pip install --user virtualenv
     ```
 
3. Pull this repository to your computer and create a new virtual environment with
     ```
     virtualenv -p python3 stat447env
     ```
  
4. Activate the environment
    OS X/Linux:
     ```
     source stat447env/bin/activate
     ```
    Windows:
     ```
     stat447env\Scripts\activate
     ```
 
5. Download [requirements.txt](requirements.txt) and put it in your working directory. Then install the dependencies listed with
     ```
     pip install -r requirements.txt
     ```
 
6. To deactivate the virtual environment, run
     ```
     deactivate
     ```
 
7. Make sure not to commit your environment folder (`.env/` I think)


[Credit](https://github.com/UBC-CS/cpsc330/blob/master/docs/setup.md)

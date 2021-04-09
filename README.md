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

# STAT 447 Project
 
Download the data [here](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_dataset_Summary.csv) and setup directory like the following
```
stat447_project
│   README.md
│   requirements.txt    
│
└───data
    │   Chest_xray_Corona_dataset_Summary.csv
    │   Chest_xray_Corona_Metadata.csv
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
		This will create a folder in your current directory that stores all the packages for this virtual environment.
	4. Activate the environment
		OS X/Linux:
		```
		source stat447env/bin/activate
		```
		Windows:
		```
		stat447env\Scripts\activate
		```
		If you happen to use csh or fish shell, source the corresponding activate file. 
		After a successful activation, something like `(cpsc330env)` should show up in the terminal.
	5. Download [requirements.txt](requirements.txt) and put it in your working directory. Then install the dependencies listed with
		```
		pip install -r requirements.txt
		```
	7. To deactivate the virtual environment, run
		```
		deactivate
  ```
 8. Make sure not to commit your environment folder (`.env/` I think)

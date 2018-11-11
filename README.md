# Predicting Time

This is my solution for a Data Science challenge, on which I had to predict the time a patient would be free after a medical consultation. The prediction should be done at the end of the triage procedure. Other details about the problem were omitted to not disclosure the challenge creator. 



# Instructions to run the code


## First things first


### Environment to run the code

I'm using Anaconda to create an environment to run the code. The following commands are "conda-related" and may differ according your SO. 

1. Run ```conda create env project```
1. Run ```activate project```
1. Run ```pip install -r requirements.txt```


### Data preparation

1. Create a folder to store the data. That folder should have, at least, the `summary.csv` file. 
1. Create a subfolder (`SUBFOLDER_NAME`), inside the data folder, to store the testing data. Put inside it the data used to evaluate the model.


## Generating the final prediction model and evaluating it 

1. Generate the final model, by running: ```python util\generate_model.py SUMMARY_DATA_FILE```. *Where*: `SUMMARY_DATA_FILE` is the fullpath for the `summary.csv` file.
1. Run ```python evaluate.py SUBFOLDER_NAME```. *Where*: `SUBFOLDER_NAME` is the folder that contains the data to evaluate the code. 





## Understanding the steps to generate the final prediction model

1. Generate the 5-folds files by running: ```python util\generate_folds.py FOLDS_FOLDER```. *Where*: `FOLDS_FOLDER` is the folder that will store the 5-fold files.
1. Analyse how different approaches perform on the data, by running:  ```python util\analyse_models.py FOLDS_FOLDER SUMMARY_DATA_FILE ```. *Where*: `FOLDS_FOLDER` is the folder with the 5-fold files, and `SUMMARY_DATA_FILE` is the fullpath for the `summary.csv` file.
1. To generate the figures/graphics of data analysis, run the command: ```python util\analyse_data.py SUMMARY_DATA_FILE ```. *Where*: `SUMMARY_DATA_FILE` is the fullpath for the `summary.csv` file.


## Generating alternative prediction modes and evaluating them
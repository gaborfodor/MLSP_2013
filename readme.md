-----------------------------------------------------------
MLSP 2013 Bird Classification Challenge -  Winning Model
Details about the competition:
http://www.kaggle.com/c/mlsp-2013-birds
-----------------------------------------------------------

I have written my model in Python 2.7.3.
I used the following libraries:
- numpy
- pandas
- scipy
- pickle
- scikit-image 
- wave
- matplotlib
- scikit-learn

Harware/OS:
Intel Core i5 2500 with 16 GB RAM 
Windows 7

--------------------
Short explanation:
--------------------
The process is splitted into four parts:
1) 1_pattern_extraction.py
2) 2a_data_preparation.py
3) 2b_data_preparation_logarithm
4) 3_train.py


The first three files contain the data preparation steps and the fourth will train RandomForestRegressors and create the submission files.

1) creates the spectrograms and does multiple the image processing steps to capture interesting patches.

2-3) After the patches are extracted the next step is feature generation using template matching. 
These are the most time consuming steps (10-12 hours) but you can run them parallel.

4) Finally I merge my features with the provided histogram of segments and location information. 
During cross validation the submission files will be exported if the cv AUC is higher than 0.93.

The current settings should produce submissions around 0.954 private leaderboard score.
A bit more about my solution can be found here: http://www.kaggle.com/c/mlsp-2013-birds/forums/t/5457/congratulations-to-the-winners/29159#post29159

-------------------------------
How to reproduce the results:
-------------------------------
Each file contains a 'folder' variable which should be manually corrected before running the code.
The easiest way is to download and extract the compressed model file which already has the required folder structure.
After the extraction you will need to copy the essential dataset into the 'essential_data' folder (mainly the audio source files and labeling informations).
The competition dataset can be downloaded from here http://www.kaggle.com/c/mlsp-2013-birds/data .
After the 'folder' variable has been modified you can start run the python sources in increasing alphabetic order. 
If you want to skip a few steps you can jump right to the training part since the 'DP' folder contains the neccessary features.
At the end you will find the resulted files in the 'Submission' folder.




import numpy as np
import pandas as pd
import scipy as sp
import  pickle
from scipy import fft
from time import localtime, strftime
import matplotlib.pyplot as plt
from skimage.feature import match_template
import wave

###########################
# Folder Name Setting
###########################
folder = 'J:/DATAMINING/KAGGLE/MLSP_BirdClassification/'

essential_folder = folder+'essential_data/'
supplemental_folder = folder+'supplemental_data/'
dp_folder = folder+'DP/'


###################################################
## Read the Essential Data 
## labels, training-test split,file_names etc.
###################################################

# Each audio file has a unique recording identifier ("rec_id"), ranging from 0 to 644. 
# The file rec_id2filename.txt indicates which wav file is associated with each rec_id.
rec2f = pd.read_csv(essential_folder + 'rec_id2filename.txt', sep = ',')

# There are 19 bird species in the dataset. species_list.txt gives each a number from 0 to 18. 
species = pd.read_csv(essential_folder + 'species_list.txt', sep = ',')
num_species = 19

# The dataset is split into training and test sets. 
# CVfolds_2.txt gives the fold for each rec_id. 0 is the training set, and 1 is the test set.
cv =  pd.read_csv(essential_folder + 'CVfolds_2.txt', sep = ',')

# This is your main label training data. For each rec_id, a set of species is listed. The format is:
# rec_id,[labels]
raw =  pd.read_csv(essential_folder + 'rec_labels_test_hidden.txt', sep = ';')
label = np.zeros(len(raw)*num_species)
label = label.reshape([len(raw),num_species])
for i in range(len(raw)):
    line = raw.irow(i)
    labels = line[0].split(',')
    labels.pop(0) # rec_id == i
    for c in labels:
        if(c != '?'):
            label[i,c] = 1

label = pd.DataFrame(label)
label['rec_id'] = cv.rec_id
label['fold'] = cv.fold
label['filename'] = rec2f.filename

# Imbalanced training set 
# training species 1%--5%--20% 
spec_avg = label[label.fold ==0][range(num_species)].mean()
spec_avg.sort()
plt.plot(spec_avg,'go')

# Read the audio files 
# /src_wavs
# This folder contains the original wav files for the dataset (both training and test sets). 
# These are 10-second mono recordings sampled at 16kHz, 16 bits per sample.

# Parameters to create the spectrogram
N = 160000
K = 512
Step = 4
wind =  0.5*(1 -np.cos(np.array(range(K))*2*np.pi/(K-1) ))
ffts = []

# wav -> np.array
def wav_to_floats(filename):
    s = wave.open(filename,'r')
    strsig = s.readframes(s.getnframes())
    y = np.fromstring(strsig, np.short)
    s.close()
    return y


###############################
## Create the Spectrograms
## Train + Test
###############################  
print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())    
for file_idx in range(len(label)):
    test_flag = label.irow(file_idx)['fold']
    fname = label.irow(file_idx)['filename']
    species_on_pic = []
    for n in range(num_species):
        if(label.irow(file_idx)[n] > 0):
            species_on_pic.append(n)
    
    S =  wav_to_floats(essential_folder+'src_wavs/'+fname+'.wav')
    Spectogram = []
    for j in range(int(Step*N/K)-Step):
        vec = S[j * K/Step : (j+Step) * K/Step] * wind
        Spectogram.append(abs(fft(vec,K)[:K/2]))
    
    ffts.append(np.array(Spectogram))

print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())

# Import the Logarithmic Patterns
pkl_file = open(dp_folder + 'LOG_SPEC_SEGMENTS.pkl', 'rb')
LOG_SPEC_SEGMENTS = pickle.load(pkl_file)
pkl_file.close() 

##################
# TRAIN FEATURES
##################
print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
TRAIN_LOG_SPEC_FEATURES = []
for file_idx in range(len(label)):
    if(file_idx %100 == 0):
        print file_idx
        print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
    
    test_flag = label.irow(file_idx)['fold']
    fname = label.irow(file_idx)['filename']
    if(test_flag == 0): # train only
        mypic = np.transpose(ffts[file_idx])
        mypic_rev = np.zeros_like(mypic)
        for i in range(mypic.shape[0]):
            mypic_rev[i] = mypic[-i - 1]
        
        mypic_rev_small = mypic_rev[:200,:] # Focus on the Relevant Frequency Domain
        mypic_rev = mypic_rev_small
        mypic_rev_log = np.log10(mypic_rev+ 0.001) # Logarithmic Transformation
        mypic_rev_log_gauss = sp.ndimage.gaussian_filter(mypic_rev_log, sigma=3) # Gaussian filter
        
        # Template Matching 
        Log_Segment_Row = []
        for s in range(len(LOG_SPEC_SEGMENTS)):
            y_min  = LOG_SPEC_SEGMENTS[s][2][0]
            y_max  = LOG_SPEC_SEGMENTS[s][2][1]
            segment = LOG_SPEC_SEGMENTS[s][3]
            if(y_min > 5):
                y_min_5 = y_min-5
            else:
                y_min_5 = 0
            
            if(y_max < 194):
                y_max_5 = y_max+5
            else:
                y_max_5 = 199
                        
            spectrogram_part = mypic_rev_log_gauss[y_min_5:y_max_5+1,:]
            result = match_template(spectrogram_part, segment)
            Log_Segment_Row.append(np.max(result))
        
        TRAIN_LOG_SPEC_FEATURES.append(Log_Segment_Row)
        

TRAIN_LOG_SPEC_FEATURES = np.array(TRAIN_LOG_SPEC_FEATURES)
    
# SAVE THE LOG-FEATURES for the TRAINING SET
output = open(dp_folder + 'TRAIN_LOG_SPEC_FEATURES_freq5.pkl', 'wb')
pickle.dump(TRAIN_LOG_SPEC_FEATURES, output)
output.close() 

print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())



###############################################################
# TEST FEATURES 
# Same Transformations and Template Matching for the Test Set
###############################################################
print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
TEST_LOG_SPEC_FEATURES = []
for file_idx in range(len(label)):
    if(file_idx %100 == 1):
        print file_idx
        print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
    
    test_flag = label.irow(file_idx)['fold']
    fname = label.irow(file_idx)['filename']
    if(test_flag == 1): # test only
        mypic = np.transpose(ffts[file_idx])
        mypic_rev = np.zeros_like(mypic)
        for i in range(mypic.shape[0]):
            mypic_rev[i] = mypic[-i - 1]
        
        mypic_rev_small = mypic_rev[:200,:]
        mypic_rev = mypic_rev_small
        mypic_rev_log = np.log10(mypic_rev+ 0.001)
        mypic_rev_log_gauss = sp.ndimage.gaussian_filter(mypic_rev_log, sigma=3)
        Log_Segment_Row = []
        for s in range(len(LOG_SPEC_SEGMENTS)):
            y_min  = LOG_SPEC_SEGMENTS[s][2][0]
            y_max  = LOG_SPEC_SEGMENTS[s][2][1]
            segment = LOG_SPEC_SEGMENTS[s][3]
            if(y_min > 5):
                y_min_5 = y_min-5
            else:
                y_min_5 = 0
            
            if(y_max < 194):
                y_max_5 = y_max+5
            else:
                y_max_5 = 199
                         
            spectrogram_part = mypic_rev_log_gauss[y_min_5:y_max_5+1,:]
            result = match_template(spectrogram_part, segment)
            Log_Segment_Row.append(np.max(result))
        
        TEST_LOG_SPEC_FEATURES.append(Log_Segment_Row)
        

TEST_LOG_SPEC_FEATURES = np.array(TEST_LOG_SPEC_FEATURES)

# SAVE THE LOG-FEATURES for the TEST SET
output = open(dp_folder + 'TEST_LOG_SPEC_FEATURES_freq5.pkl', 'wb')
pickle.dump(TEST_LOG_SPEC_FEATURES, output)
output.close() 
print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())




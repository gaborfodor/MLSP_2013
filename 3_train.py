
import numpy as np
import pandas as pd
import  pickle
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics

###########################
# Folder Name Setting
###########################
folder = 'J:/DATAMINING/KAGGLE/MLSP_BirdClassification/'


essential_folder = folder+'essential_data/'
supplemental_folder = folder+'supplemental_data/'
dp_folder = folder+'DP/'
subm_folder = folder+ 'Submission/'
log_folder = folder+ 'log/'


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

spec_avg = label[label.fold ==0][range(num_species)].mean()
plt.plot(spec_avg,'go')
plt.plot(-np.log(spec_avg),'bo')
spec_num_features = -np.log(spec_avg)


hos =  pd.read_csv(supplemental_folder + 'histogram_of_segments.txt', sep = ',',skiprows=1,header=0)
hos_features = ['hos_'+str(x) for x in range(100) ]
hos.columns = ['rec_id'] +hos_features

data = pd.merge(left = label, right = hos ,how = 'left', left_on = 'rec_id', right_on = 'rec_id')
data = data.fillna(0)

##############################
##          TRAIN           ##
##############################
train = data[data.fold==0]
pc = []
for i  in range(len(train)):
    s = train.filename.irow(i)[:5]
    pc.append(int(s[2:s.find('_')] )) # PC1 - PC18

train['pc'] = pc

 
pkl_file = open(dp_folder + 'TRAIN_SPEC_FEATURES_freq5.pkl', 'rb')
tr_spec = pickle.load(pkl_file)
pkl_file.close()
spec_names = ['tr_spec_'+str(x) for x in range(tr_spec.shape[1]) ]
Spec_Df = pd.DataFrame(tr_spec,columns = spec_names )
Spec_Df['rec_id']= train.index
train2 = pd.merge(left = train, right = Spec_Df , left_index = True, right_on = 'rec_id')

pkl_file = open(dp_folder + 'TRAIN_LOG_SPEC_FEATURES_freq5.pkl', 'rb')
tr_log_spec = pickle.load(pkl_file)
pkl_file.close()
log_spec_names = ['tr_log_spec_'+str(x) for x in range(tr_log_spec.shape[1]) ]
Spec_Log_Df = pd.DataFrame(tr_log_spec,columns = log_spec_names )
Spec_Log_Df['rec_id']= train.index

train3 = pd.merge(left = train2, right = Spec_Log_Df , left_on = 'rec_id', right_on = 'rec_id')

##############################
##          TEST            ##
##############################
test = data[data.fold==1]
pc = []
for i  in range(len(test)):
    s = test.filename.irow(i)[:5]
    pc.append(int(s[2:s.find('_')] )) # PC1 - PC18

test['pc'] = pc

pkl_file = open(dp_folder + 'TEST_SPEC_FEATURES_freq5.pkl', 'rb')
test_spec = pickle.load(pkl_file)
pkl_file.close()
Test_Spec_Df = pd.DataFrame(test_spec,columns = spec_names )
Test_Spec_Df['rec_id']= test.index
test2 = pd.merge(left = test, right = Test_Spec_Df , left_index = True, right_on = 'rec_id')

pkl_file = open(dp_folder + 'TEST_LOG_SPEC_FEATURES_freq5.pkl', 'rb')
test_log_spec = pickle.load(pkl_file)
pkl_file.close()
Test_Spec_Log_Df = pd.DataFrame(test_log_spec,columns = log_spec_names )
Test_Spec_Log_Df['rec_id']= test.index

test3 = pd.merge(left = test2, right = Test_Spec_Log_Df , left_on = 'rec_id', right_on = 'rec_id')


#######################################################
## PARAMETER OPTIMIZATION & SUBMISSION CREATION      ##
#######################################################

CV_FOLDS = 15

RESULT = []
rs = 0
for ID in range(1):    
    for NUM_FEATURES in range(40,50,10):
        for N_ESTIMATORS in range(500,501,100):
            for MAX_FEATURES in range(4,5):
                for MIN_SAMPLES_SPLIT in range(2,3):
                    cv =  np.random.randint(0,CV_FOLDS,len(train))
                    train3['cv'] = cv
                    labeled_vector = []
                    predicted_vector = []
                    predicted_test_vector = []
                    for bird in range(num_species):
                        predicted_test_vector.append(np.zeros(len(test3)))
                        
                    for c in range(CV_FOLDS):
                        df_10 = train3[train3.cv == c]
                        df_90 = train3[train3.cv != c]
                        X_90 = df_90[spec_names+hos_features+['pc']+log_spec_names]
                        X_10 = df_10[spec_names+hos_features+['pc']+log_spec_names]
                        T = test3[spec_names+hos_features+['pc']+log_spec_names]
                        for bird in range(num_species):
                            rs = rs+1
                            y_90 = df_90[bird]
                            y_10 = df_10[bird]
                            selector = SelectKBest(f_regression,NUM_FEATURES + 50 -int(spec_num_features[bird]*10))
                            selector.fit(X_90, y_90)
                            df_90_features = selector.transform(X_90)
                            df_10_features = selector.transform(X_10)
                            T_features = selector.transform(T)
                            rfr = RandomForestRegressor(n_estimators = N_ESTIMATORS, max_features = MAX_FEATURES, min_samples_split = MIN_SAMPLES_SPLIT,random_state = rs*100, verbose = 0)
                            rfr.fit(df_90_features,y_90)
                            p_10 = rfr.predict(df_10_features)
                            T_pred = rfr.predict(T_features)
                            predicted_vector = predicted_vector + list(p_10)
                            labeled_vector = labeled_vector + list(y_10)
                            predicted_test_vector[bird] = predicted_test_vector[bird] + T_pred/CV_FOLDS
                    
                    fpr, tpr, thresholds = metrics.roc_curve(labeled_vector, predicted_vector, pos_label=1)
                    auc = metrics.auc(fpr,tpr)
                    
                    RESULT.append([ID,NUM_FEATURES,N_ESTIMATORS,MAX_FEATURES,MIN_SAMPLES_SPLIT,CV_FOLDS,auc])
                    ResultDf = pd.DataFrame(RESULT,columns=['ID','NUM_FEATURES','N_ESTIMATORS','MAX_FEATURES','MIN_SAMPLES_SPLIT','CV_FOLDS','AUC'])
                    ResultDf.to_csv(log_folder +'rfr_auc_result.txt', index = False)
                    
                    if(auc > 0.93):
                        Submission_ID = []
                        Submission_PROB = []
                        for bird in range(num_species):
                            ids = np.array(test3.rec_id) *100 + bird
                            probs = predicted_test_vector[bird]
                            Submission_ID = Submission_ID + list(ids)
                            Submission_PROB = Submission_PROB + list(probs)
                        
                        SubmissionDf = pd.DataFrame(Submission_ID,columns=['Id'])
                        SubmissionDf['Probability'] = Submission_PROB
                        SubmissionDf.to_csv(subm_folder+'rfr_osl_cv15_freq_5'+str(ID)+'_'+str(NUM_FEATURES)+'_'+str(N_ESTIMATORS)+'_'+str(MAX_FEATURES)+'_'+str(MIN_SAMPLES_SPLIT)+'_'+str(CV_FOLDS)+'.csv', index = False)


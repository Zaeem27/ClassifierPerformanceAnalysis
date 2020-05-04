import os
import os.path
from os import path
import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

rootdir = 'D:/2019-20/Winter/4313/EECS4313_Proj/data/xorg'
file = open(rootdir+"/tables.txt", "w")

for subdir, dirs, files in os.walk(rootdir):
    if path.exists(subdir+"/test.csv") and path.exists(subdir+"/train.csv"):     
        #print os.path.join(subdir)
        print("Processing " + subdir)
        file.write(subdir +"\n")
        input_file_training = subdir+"/train.csv"
        input_file_test = subdir+"/test.csv"
        
        # load the training data as a matrix
        dataset = pd.read_csv(input_file_training, header=0)
        dataset.rename(columns={"500_Buggy?": "Buggy"}, inplace = True)

        # separate the data from the target attributes
        train_data = dataset.drop('change_id', axis=1)
        train_data = train_data.drop('412_full_path', axis=1)
        train_data = train_data.drop('411_commit_time', axis=1)
        # remove unnecessary features
        #train_data = train_data.drop('File', axis=1)

        # the lables of training data. label is the title of the  last column in your CSV files
        train_target = dataset.Buggy

        # load the testing data
        dataset2 = pd.read_csv(input_file_test, header=0)
        dataset2.rename(columns={"500_Buggy?": "Buggy"}, inplace = True)
        
        # separate the data from the target attributes
        test_data = dataset2.drop('change_id', axis=1)
        test_data = test_data.drop('411_commit_time', axis=1)
        test_data = test_data.drop('412_full_path', axis=1)

        # remove unnecessary features
        #test_data = test_data.drop('File', axis=1)

        # the lables of test data
        test_target = dataset2.Buggy

        #print(test_target)


        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(train_data, train_target)
        test_data_resampled, test_target_resampled = rus.fit_resample(test_data, test_target)
        gnb = GaussianNB()
        #decision_t = DecisionTreeClassifier(random_state=0, max_depth=2)
        test_pred = gnb.fit(X_resampled, y_resampled).predict(test_data_resampled)

        file.write(classification_report(test_target_resampled, test_pred, labels=[0,1]))
        file.write("\n")
file.close() 

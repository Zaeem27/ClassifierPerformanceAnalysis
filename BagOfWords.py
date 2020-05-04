from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
import csv
import io
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

#########This whole thing is trying to build the bag of words myself##########
rootdir = 'D:/2019-20/Winter/4313/EECS4313_Proj/data/jdt'
patchdir = rootdir + "/patch"
excludedwords = ["int", "double", "boolean", "float", "class", "public", "void", "private"]
word_dict ={}
selected_word_list=[]
print("Building word dictionary")
for filename in os.listdir(patchdir):
    if filename.endswith(".patch"):
         with io.open(os.path.join(patchdir, filename), encoding='latin-1') as f:
             for line in f:
                for word in line.split():
                    if word not in excludedwords:
                        if word in word_dict:
                            word_dict[word] +=1
                        else:
                            word_dict[word] = 1

print("Pruning dictionary")
for k,v in word_dict.items():
    if v > 20:
       selected_word_list.append(k)
##################################################################

    
################This uses sklearn libraries to try to build the bag of words##########################    
for subdir, dirs, files in os.walk(rootdir):
    if path.exists(subdir+"/test.csv") and path.exists(subdir+"/train.csv"):     
     input_file_training = subdir+"/train.csv"
     input_file_test = subdir+"/test.csv"
     dataset = pd.read_csv(input_file_training, header=0)
     dataset.rename(columns={"500_Buggy?": "Buggy"}, inplace = True)
     change_ids = dataset.change_id
     bugs = dataset.Buggy
     f = open (subdir+'/BOW_Features_Train.csv', 'w')
     wtr = csv.writer(f, delimiter=',', lineterminator='\n')
     wtr.writerow(["for","while","if", "then", "else", "switch", "case", "do", "Buggy"])
     i=0
     for ids in change_ids:
         #file = open(rootdir+"/patch/jdt-" + str(ids) + ".patch", "r")
         file = io.open(rootdir+"/patch/jdt-" + str(ids) + ".patch", encoding='latin-1')
         print(file.name)
         file_text = (file.read())
         file_text=[file_text]
         count_vect = CountVectorizer(min_df=3, vocabulary=["for","while","if", "then", "else", "switch", "case", "do"])
         #count_vect = CountVectorizer(vocabulary=selected_word_list)
         X_train_counts = count_vect.fit_transform(file_text)
         freq_array = (X_train_counts.toarray())
         freq_array = np.append(freq_array, [bugs[i]])
         freq_array=[freq_array]
         i=i+1
         for x in freq_array:
             wtr.writerow(x)
     f.close()
     dataset = pd.read_csv(input_file_test, header=0)
     dataset.rename(columns={"500_Buggy?": "Buggy"}, inplace = True)
     change_ids = dataset.change_id
     bugs = dataset.Buggy
     f = open (subdir+'/BOW_Features_Test.csv', 'w')
     wtr = csv.writer(f, delimiter=',', lineterminator='\n')
     wtr.writerow(["for","while","if", "then", "else", "switch", "case", "do", "Buggy"])
     i=0
     for ids in change_ids:
         #file = open(rootdir+"/patch/jdt-" + str(ids) + ".patch", "r")
         file = io.open(rootdir+"/patch/jdt-" + str(ids) + ".patch", encoding='latin-1')
         print(file.name)
         file_text = (file.read())
         file_text=[file_text]
         count_vect = CountVectorizer(min_df=3, vocabulary=["for","while","if", "then", "else", "switch", "case", "do"])
         #count_vect = CountVectorizer(vocabulary=selected_word_list)
         X_train_counts = count_vect.fit_transform(file_text)
         freq_array = (X_train_counts.toarray())
         freq_array = np.append(freq_array, [bugs[i]])
         freq_array=[freq_array]
         i=i+1
         for x in freq_array:
             wtr.writerow(x)
     f.close()

print("Building model")
tableFile = open(rootdir+"/tables.txt", "w")
for subdir, dirs, files in os.walk(rootdir):
    if path.exists(subdir+"/test.csv") and path.exists(subdir+"/train.csv"):     
        #print os.path.join(subdir)
        print("Processing " + subdir)
        tableFile.write(subdir +"\n")
        input_file_training = subdir+"/BOW_Features_Train.csv"
        input_file_test = subdir+"/BOW_Features_Test.csv"
        
        # load the training data as a matrix
        dataset = pd.read_csv(input_file_training, header=0)
        

        # separate the data from the target attributes
        train_data = dataset

        # the lables of training data. label is the title of the  last column in your CSV files
        train_target = dataset.Buggy

        # load the testing data
        dataset2 = pd.read_csv(input_file_test, header=0)
       
        # separate the data from the target attributes
        test_data = dataset2

        # the lables of test data
        test_target = dataset2.Buggy

        #print(test_target)

        gnb = GaussianNB()
        #clf= ExtraTreeClassifier(splitter='best')
        #clf = svm.SVC(gamma='scale', class_weight='balanced')
        #decision_t = DecisionTreeClassifier(random_state=0, max_depth=2)
        #regr = RandomForestRegressor(max_features=None,max_depth=None,n_estimators=10000)
        clf = LogisticRegression(warm_start=True,max_iter=1000000000)

        test_pred = clf.fit(train_data, train_target).predict(test_data)

        tableFile.write(classification_report(test_target, test_pred, labels=[0,1]))
        tableFile.write("\n")
tableFile.close() 

    










# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:31:20 2019

@author: Victor Zuanazzi
"""

#base libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
import warnings
import time

#advanced libaries
import pickle
import category_encoders as ce
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, Imputer
import lime
from lime import lime_tabular
#rom treeinterpreter import treeinterpreter as ti

#ML libaries
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, datasets, tree, svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, classification_report
from sklearn.externals import joblib
#from sklearn import cross_validation
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

#DL libaries
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import model_from_json
import keras.callbacks as cbks
from keras.constraints import min_max_norm
from keras.constraints import unit_norm

#Progress bars
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm 
tqdm.pandas()


def runLearningModel(x_train, x_test, y_train, y_test, model, log = False):
    if(model=="LOG"):
        MOD = linear_model.LogisticRegression()
    if(model=="GAUSS"):
        MOD = GaussianNB()
    if(model=="TREE"):
        MOD = tree.DecisionTreeClassifier()
    if(model=="NN"):
        MOD = MLPClassifier(hidden_layer_sizes=(5, 5),solver='sgd')
    if(model=="RF"): #Best Performer so far
        mf = np.array(x_train).shape[1]
        MOD = RandomForestClassifier(max_features = mf, #original: 41, 
                                   n_estimators = 10, 
                                   max_depth = 15, 
                                   min_samples_split = 3, 
                                   verbose=2) 
    
    if(model=="SVM"): #Slow, but the best for recall
        MOD = svm.SVC(kernel = 'linear',probability=True, max_iter=1000)
        #max_iter = 1000 is a good number for a coffee
            
    MOD.fit(x_train, y_train.astype("int"))
    y_pred_class = MOD.predict(x_test)
    y_pred_prob =  MOD.predict_proba(x_test)
    if log:
        print(model,"\n confusion matrix:")
        print(confusion_matrix(y_test.astype("int"), y_pred_class))
    accuracy = accuracy_score(y_test.astype("int"), y_pred_class)
    precision = 1 # precision_score(y_test.astype("int"), y_pred_class, average = "samples")
    recall = 1 #recall_score(y_test.astype("int"), y_pred_class)
    
    return [accuracy,precision, recall], y_pred_class, y_pred_prob, MOD


def loadData(path_x = './data/Splits.csv', path_y = './data/SplitsLabel.csv'):  
    """
    """  
    #if everything works fine, the ith line of df_y is the label for the ith 
    #line of df_x
#    df = pd.read_csv(filepath_or_buffer = path_x,
#                       header = None, 
#                       names = ["i", "clauses"]) 
#    
#    #it is sad that I have to do that!
#    df["clauses"] = df["clauses"].apply(lambda x: x.replace("[", ""))
#    df["clauses"] = df["clauses"].apply(lambda x: x.replace("]", ""))
#    df["clauses"] = df["clauses"].apply(lambda x: x.replace("\n", ""))
#    df["clauses"] = df["clauses"].apply(lambda x: x.replace(",", ""))
#    df["clauses"] = df["clauses"].apply(lambda x: x.split(" "))
#    df["clauses"] = df["clauses"].apply(lambda x: np.array([int(i) for i in x]))
#    
    df_x = pd.read_csv(filepath_or_buffer = path_x,
                       header = None, 
                       names =  [x for x in range(120060)]) 
    
#    cols = [x for x in range(120060)]
#    df_x = pd.DataFrame(columns = cols)
#    
#    for i in range(len(df)):   
#        df_x = df_x.append(pd.Series(data = {k: df["clauses"][i][k] for k in cols}),
#                           ignore_index = True)
    
#    df = pd.read_csv(filepath_or_buffer = path_y,
#                       header = None, 
#                       names = ["i", "label"]) 
#    
#    df_x.astype("int")
#    
#    #it is sad that I have to do that!
#    df["label"] = df["label"].apply(lambda x: x.replace("[", ""))
#    df["label"] = df["label"].apply(lambda x: x.replace("]", ""))
#    df["label"] = df["label"].apply(lambda x: x.replace("\n", ""))
#    df["label"] = df["label"].apply(lambda x: x.replace(",", ""))
#    df["label"] = df["label"].apply(lambda x: x.split(" "))
#    df["label"] = df["label"].apply(lambda x: [int(i) for i in x])
#    
#    cols = [y for y in range(81)]
#    df_y = pd.DataFrame(columns = cols)
#    
#    for i in range(len(df)):   
#        df_y = df_y.append(pd.Series(data = {k: df["label"][i][k] for k in cols}),
#                           ignore_index = True)
    df_y = pd.read_csv(filepath_or_buffer = path_y,
                       header = None, 
                       names =  [x for x in range(81)]) 
    df_y.astype("int")
    
    #sanety check:
    if len(df_x) < len(df_y):
        print(f"WARNING: possible missmatch between df_x: {len(df_x)} and df_y: {len(df_y)}")
        df_y = df_y[0:len(df_x)]
    elif len(df_x) > len(df_y):
        print(f"WARNING: possible missmatch between df_x: {len(df_x)} and df_y: {len(df_y)}")
        df_x = df_x[0:len(df_y)]        
    
    return df_x, df_y

def dataSplit(df_x, df_y):
    """split data into test and train """
    x_train, x_test, y_train, y_test = train_test_split(df_x, 
                                                        df_y, 
                                                        test_size = 0.2, 
                                                        random_state = 42)
    
    return x_train, x_test, y_train, y_test

def trainModel(model, df_x, df_y):
    
    x_train, x_test, y_train, y_test = dataSplit(df_x, df_y[0])
                                       
    results, y_test_pred_class, y_test_pred_prob, MOD = runLearningModel(x_train,
                                                                         x_test,
                                                                         y_train, 
                                                                         y_test, 
                                                                         model, 
                                                                         log = False)
    return results, MOD        

def main():
    #models = ['RF', 'LOG', 'GAUSS', 'TREE', 'NN', 'SVM']
    model = "RF"
    
    df_x, df_y = loadData()
    results, MOD = trainModel(model, df_x, df_y)
    print("results: ", results)
    print("Model: ", MOD)
           
            
if __name__ == "__main__":
    main() 
            
            
            
            


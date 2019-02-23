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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

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


#feature size reduced for memory sake...
NUM_CLAUSES = 100
l_PER_CLAUSE = 4

def score_function(estimator, X, y):
    
    y_pred_class = estimator.predict(X)
    
    acc = y_pred_class == y
    
    return sum(acc)
    

def runLearningModel(x_train, x_test, y_train, y_test, model, log = False):
    
    if(model=="RF"): #Best Performer so far
        MOD = RandomForestClassifier()
    elif(model == "NN"):
        
        MOD = MLPClassifier((100,))
    
    #x_train, x_test, y_train, y_test = dataSplit(df_x, df_y)
        
    m_params = { 
            "RF": {
                    "n_estimators" : [1, 2],    #worth replacing with a distribution
                    "max_depth": [2, 3],         #worth replacing with a distribution
                    "min_samples_split": [2, 10],  #worth replacing with a distribution
                    "max_features": ["sqrt", None],
                    "verbose": [2]
                    },
            "NN": {
                    "hidden_layer_sizes": [(100), (200, 200)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "learning_rate_init": [0.001, 0.0001], #worth replacing with a distribution
                    "max_iter": [200, 500], #worth replacing with a distribution
                    }
            }
    
     
    search_inter = 2
    random_search = RandomizedSearchCV(MOD,
                                       param_distributions = m_params[model], 
                                       n_iter = search_inter,
                                       scoring = score_function,
                                       return_train_score = True) 
    
    random_search.fit(x_train, y_train)
   
    y_pred_class = random_search.best_estimator_.predict(x_test)
    y_pred_prob =  random_search.best_estimator_.predict_proba(x_test)
    acc = y_pred_class == y_test
    accuracy = acc.sum(axis = 1)/len(y_test)
    print("Acuracy:", accuracy)
    
    if log:
        print(model,"\n confusion matrix:")
        print(confusion_matrix(y_test.astype("int"), y_pred_class))
    
    return accuracy, y_pred_class, y_pred_prob, random_search.best_estimator_


def loadData(path_x = './data/Splits.csv', path_y = './data/SplitsLabel.csv'):  
    """
    """      
    #TO-DO: find a more compact feature representation.
    
    #if everything works fine, the ith line of df_y is the label for the ith 

    
    print('Loading data...', end='\r')    
    df_x = pd.read_csv(filepath_or_buffer = path_x,
                       header = None, 
                       names =  [x for x in range(NUM_CLAUSES*l_PER_CLAUSE)]) 
    
    print('Loading labels...', end='\r')
    df_y = pd.read_csv(filepath_or_buffer = path_y,
                       header = None, 
                       names =  [x for x in range(81)]) 
    
    #Excludes the sudokus that were not solved
    print('Pre-processing...', end='\r')
    nan_idx = df_y[1].index[df_y[1].apply(np.isnan)] #
    df_y.drop(labels = nan_idx, axis = 0, inplace = True)
    df_x.drop(labels = nan_idx, axis = 0, inplace = True)
    
    #Everything is an int.
    df_y.astype("int")
    df_x.astype("int")
    
    print('Done!')
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
    
    x_train, x_test, y_train, y_test = dataSplit(df_x, df_y)
                                       
    results, y_test_pred_class, y_test_pred_prob, MOD = runLearningModel(x_train,
                                                                         x_test,
                                                                         y_train, 
                                                                         y_test, 
                                                                         model, 
                                                                         log = False)
    
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(MOD, open(filename, 'wb'))
    
    return results, MOD   

def learnedHeuristic(cnf):  
    """returns a literal by a learned heuristics
    """
    
    #tranlates cnf to a array the model understands    
    clauses = np.zeros((NUM_CLAUSES, l_PER_CLAUSE)) 
    
    for c, clause in enumerate(cnf[:NUM_CLAUSES]):
        for l, literal in enumerate(list(clause.keys())[:(l_PER_CLAUSE-1)]):
           clauses[c][l] = literal 
           
    clauses = np.reshape(clauses, (1, NUM_CLAUSES*l_PER_CLAUSE))
    
    filename = 'finalized_model.sav'
    MOD = pickle.load(open(filename, 'rb'))
    smart_literal = MOD.predict(clauses)[0]
    
    count = 0
    for sl in smart_literal:
        for c in cnf:
            count += 1
            if sl in list(c.keys()):
                print("learned")
                return sl, True
            if count > 100:
                print("this one instead")
                return list(c.keys())[0], True
        
    print("next instead")
    return list(cnf[0].keys())[0], True

def main():
    #models = ['RF', 'NN']
    model = "RF"
    
    df_x, df_y = loadData()
    #y_enc = MultiLabelBinarizer().fit_transform(df_y)
    results, MOD = trainModel(model, df_x, df_y)
    print("results: ", results)
    print("Model: ", MOD)
           
            
if __name__ == "__main__":
    main() 
            
            
            
            


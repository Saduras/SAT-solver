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
import warnings
import time

#advanced libaries
import pickle
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

from heuristics import nextLiteral

#Progress bars
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm 
tqdm.pandas()


#feature size reduced for memory sake...
NUM_CLAUSES = 100
l_PER_CLAUSE = 4

def scoreFunction(estimator, X, y):
    """models will be trainded to maximize the scoreFunction.
    It sums all the correct assignments.
    Inputs:
        estimator: a sklearn model with a .predict routine.
        X: matrix of features
        y: matrix of labels
        
    Returns: sum of all correnct assignments.
    """
    
    #predict classes
    y_pred_class = estimator.predict(X)
    
    #create a matrix of correct assingments.
    acc = y_pred_class == y
    
    #sum all correct assingments.
    return sum(acc)
    

def runLearningModel(x_train, x_test, y_train, y_test, model, log = False):
    """Create and optimize a model given the training data.
    Input:
        x_train: feature matrix for training.
        x_test: feature matrix to gather statistics about the model performance 
            after it was trained and optimized.
        y_train: label matrix for training.
        y_test: label matrix to gather statistics about the model performance 
            after it was trained and optimized.
        model: "RF" is implemented and tested.
        log: whether to print final results or not and whether to give more 
            detailed log of the progress.
    Output:
        accuracy: per line of the test matrices, % of classes the model 
            predicted correctly.
        y_pred_class: predicted classes for the given x_test
        y_pred_prob: the confidence of each prediction.
        
    """
    
    #Implemented models:
    if(model=="RF"): #Best Performer so far
        MOD = RandomForestClassifier()
    elif(model == "NN"):      
        MOD = MLPClassifier() #not working yet
     
    #dict with dicts with parameters to be optimized.    
    m_params = { 
            "RF": {
                    "n_estimators" : np.linspace(10, 100, 30, dtype = "int"),    #worth replacing with a distribution
                    "max_depth": [10, 20, 50, 100, None],         #worth replacing with a distribution
                    "min_samples_split": np.linspace(2, 10, 5, dtype = "int"),  #worth replacing with a distribution
                    "max_features": ["sqrt", "log2", None],
                    "verbose": [1+int(log)]
                    },
            "NN": {
                    "hidden_layer_sizes": [(100), (200, 200)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "learning_rate_init": [0.001, 0.0001], #worth replacing with a distribution
                    "max_iter": [200, 500], #worth replacing with a distribution
                    }
            }
    
    #number of iterations for optimization. 
    search_inter = 10
    random_search = RandomizedSearchCV(MOD,
                                       param_distributions = m_params[model], 
                                       n_iter = search_inter,
                                       scoring = scoreFunction,
                                       return_train_score = True,
                                       random_state = 42,
                                       cv = 5) 
    
    #optimization + train the model.
    random_search.fit(x_train, y_train)
    
    #recover the best model
    opt_model = random_search.best_estimator_
    
    #gather performace metrics.
    y_pred_class = opt_model.predict(x_test)
    y_pred_prob =  opt_model.predict_proba(x_test)
    acc = y_pred_class == y_test  
    accuracy = acc.sum(axis = 1)/81

    if log:
        print(acc)
        print("Mean accuracy:", accuracy.mean())

    return accuracy, y_pred_class, y_pred_prob, opt_model


def loadData(path_x = './data/SSplits.csv', path_y = './data/SplitsLabel.csv'):  
    """load data for training the model.
    Must be in format x,81
    Input:
        path_x: (str), file path to feature matrix.
        path_y: (str), file path to label matrix.
    
    Output:
        df_x: (pd.DataFrame) feature matrix loaded in memory.
        df_y: (pd.DataFrame) label matrix loaded in memory.
    """    
 
    #if everything works fine, the ith line of df_y is the label for the ith 
    print('Loading data...', end='\r')    
    df_x = pd.read_csv(filepath_or_buffer = path_x,
                       header = None, 
                       names = [x for x in range(81)]) #names =  [x for x in range(NUM_CLAUSES*l_PER_CLAUSE)]) 
    
    print('Loading labels...', end='\r')
    df_y = pd.read_csv(filepath_or_buffer = path_y,
                       header = None, 
                       names =  [x for x in range(81)]) 
    
    #Excludes the sudokus that were not solved
    print('Pre-processing...', end='\r')
    #nan_idx = df_y[1].index[df_y[1].apply(np.isnan)] #
    #df_y.drop(labels = nan_idx, axis = 0, inplace = True)
    #df_x.drop(labels = nan_idx, axis = 0, inplace = True)
    
    #Everything is an int.
    df_y.astype("int")
    df_x.astype("int")
    
    print('Done!')
    
    #safety check:
    if len(df_x) < len(df_y):
        print(f"WARNING: possible missmatch between df_x: {len(df_x)} and df_y: {len(df_y)}")
        df_y = df_y[0:len(df_x)]
    elif len(df_x) > len(df_y):
        print(f"WARNING: possible missmatch between df_x: {len(df_x)} and df_y: {len(df_y)}")
        df_x = df_x[0:len(df_y)]        
    
    return df_x, df_y

def dataSplit(df_x, df_y):
    """split data into test and train
    input:
        df_x: (pd.DataFrame) feature matrix
        df_y: (pd.DataFrame) label matrix
    
    Output:
        x_train: (pd.DataFrame) feature matrix for training the model.
        x_test: (pd.DataFrame) feature matrix for testing the model.
        y_train: (pd.DataFrame) label matrix for training the model.
        y_test: (pd.DataFrame) label matrix for testing the model.
    """
    
    x_train, x_test, y_train, y_test = train_test_split(df_x, 
                                                        df_y, 
                                                        test_size = 0.2, 
                                                        random_state = 42)
    
    return x_train, x_test, y_train, y_test

def trainModel(model, df_x, df_y):
    """It splits the data into test and train.
    It only trains the model in the train data.
    Input:
        model: (str) key of one of the implemented models at runLearningModel().
        df_x: (pd.DataFrame) feature matrix.
        df_y: (pd.DataFrame) label matrix.
        
    Output:
        results: (list(floats)) % of classes the model did correctly for the 
            test data.
        MOD: the optimized model.
    """
    
    #split the data
    x_train, x_test, y_train, y_test = dataSplit(df_x, df_y)
    
    #maybe some it would be interesting to normalize the data. 
      
    #otimize and train the model.                                 
    results, y_test_pred_class, y_test_pred_prob, MOD = runLearningModel(x_train,
                                                                         x_test,
                                                                         y_train, 
                                                                         y_test, 
                                                                         model, 
                                                                         log = False)
    
    # save the model to disk
    filename = model + 'finalized_model.sav'
    pickle.dump(MOD, open(filename, 'wb'))
    
    return results, MOD   


def learnedHeuristic(cnf, assignment, model = "RF"):
    """given a sudoku assignment, the model finds a possible solution.
    Input:
        assigment: (list(int)) current assigments for the sudoku.
    Output:
        one literal for the split.
    """
    
    #mapping from sudoku literal to cell space at current_sudoku.
    ass2sud = {x + (i+1)*10: i for i in range(81) for x in range(101, 190)}
    
    #loads the state of the sudoku.
    current_sudoku = np.zeros((1, 81))
    for a in assignment: #is there a pythonic way of writing it?
        if a > 0:
            current_sudoku[0][ass2sud[a]] = a
    
    #positions of the sudoku that still have to be resolved.
    _, open_ass = np.where(current_sudoku == 0)
    
    #load saved model.    
    filename = model + 'finalized_model.sav'
    MOD = pickle.load(open(filename, 'rb'))
    
    #get model prediction
    smart_literal = MOD.predict(current_sudoku)[0]

    
    if np.random.uniform() > .5: #necessary to avoid infinite loops.   
        #return one of the predicted literals for a unresolved cell of the
        #sudoku.
        sl = np.random.choice(open_ass)
        print("learned:", smart_literal[sl] )
        return int(smart_literal[sl]), True
    
    else:   
        #calls next literal
        a =  nextLiteral(cnf)  
        print("NEXT LITERAL:", a)
        return  a

def learnedHeuristicCNF(cnf):  
    """deprecated.
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
    results, MOD = trainModel(model, df_x, df_y)
    print("results: ", results)
    print("mean accuracy:", results.mean())
    print("Model: ", MOD)
           
            
if __name__ == "__main__":
    main() 
            
            
            
            


# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:31:20 2019

@author: Victor Zuanazzi
"""

#base libraries
import numpy as np
import pandas as pd

#advanced libraries
import pickle

#ML librariesGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import random


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
    
    #only sum the assignments where the sudoku is fully correct
    accuracy = acc.sum(axis = 1)/81 == 1
    
    #sum all fully correct assingments.
    return accuracy.mean()
    

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
                    "n_estimators" : np.linspace(2, 50, 50, dtype = "int"),    #worth replacing with a distribution
                    "max_depth": [5, 10, None],         #worth replacing with a distribution
                    "min_samples_split": np.linspace(2, 10, 5, dtype = "int"),  #worth replacing with a distribution
                    "max_features": ["sqrt", "log2", 21 , 42 ,None],
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
    random_search = RandomizedSearchCV(MOD,
                                       param_distributions = m_params[model], 
                                       n_iter = 20,
                                       scoring = scoreFunction,
                                       return_train_score = True,
                                       random_state = 42,
                                       cv = 3) 
    
    #trains and optimizes the model
    random_search.fit(x_train, y_train)
    
    #recover the best model
    opt_model = random_search.best_estimator_
    
    #gather performace metrics.
    y_pred_class = opt_model.predict(x_test)
    y_pred_prob =  opt_model.predict_proba(x_test)
    acc = y_pred_class == y_test  
    accuracy = acc.sum(axis = 1)/81 == 1

    if log:
        print(acc)
        print("Mean accuracy:", accuracy.mean())

    return accuracy, y_pred_class, y_pred_prob, opt_model


def loadData(path_x = '../data/SSplits.csv', path_y = '../data/SplitsLabel.csv'):  
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
                       names = [x for x in range(81)]) 
                      
    print('Loading labels...', end='\r')
    df_y = pd.read_csv(filepath_or_buffer = path_y,
                       header = None, 
                       names =  [x for x in range(81)]) 
    
    
    print('Pre-processing...', end='\r')
    
    #Ensures correct indexing:
    df_x.reset_index(inplace = True, drop = True)
    df_y.reset_index(inplace = True, drop = True)
    
    #safety check:
    if len(df_x) < len(df_y):
        print(f"WARNING: possible missmatch between df_x: {len(df_x)} and df_y: {len(df_y)}")
        df_y = df_y[0:len(df_x)]
    elif len(df_x) > len(df_y):
        print(f"WARNING: possible missmatch between df_x: {len(df_x)} and df_y: {len(df_y)}")
        df_x = df_x[0:len(df_y)]    
    
    #Excludes the sudokus that were not solved
    nan_idx = df_y[1].index[df_y[1].apply(np.isnan)] 
    if len(nan_idx) > 0:
        df_x.drop(labels = nan_idx, axis = 0, inplace = True)
        df_y.drop(labels = nan_idx, axis = 0, inplace = True)
    
    #strips the cordinates away:
    df_x = df_x % 10
    df_y = df_y % 10
     
    #Everything is an int.
    df_y.astype("int")
    df_x.astype("int")
    
    print('Done!')
    
           
    
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
    
    #filename = model + 'finalized_model.sav'
    #pickle.dump(MOD, open(filename, 'wb')) 
    
    #save model to disk 
    #the saved file is 200Mb+ 
    filename = "../models/" + model + 'finalized_model.sav'
    pickle.dump(MOD, open(filename, 'w+b'))  
    
    return results, MOD   


def learnedHeuristic(cnf, assignment, model = "RF"):
    """given a sudoku assignment, the model finds a possible solution.
    Input:
        assigment: (list(int)) current assigments for the sudoku.
    Output:
        one literal for the split.
    """
    
    #loads the state of the sudoku.
    current_sudoku = np.zeros((9, 9))
    for x in assignment:
        if x  <= 0:
            continue 
        #extract coordinates:
        i = int(x/100) - 1 
        j = int(x/10)%10 - 1
        current_sudoku[i][j] = x%10
    current_sudoku = np.reshape(current_sudoku, (1,81))
    
    #positions of the sudoku that still have to be resolved.
    _, open_ass = np.where(current_sudoku == 0)
    
    #load saved model.    
    filename = "..//models//" + model + 'finalized_model.sav'
    MOD = pickle.load(open(filename, 'rb'))
    
    #get model prediction
    MOD.verbose = False
    smart_literal = MOD.predict(current_sudoku)[0].astype("int")
    #literal_prob =  MOD.predict_proba(current_sudoku)
      
    #includes the coordinates back in the numbers.
    smart_literal = np.reshape(smart_literal, (9,9))
    for i in range(len(smart_literal)):
       for j in range(len(smart_literal[0])):
           smart_literal[i][j] += (i+1)*100 + (j+1)*10       
    smart_literal = np.reshape(smart_literal, (1,81))    
           
    #return one of the predicted literals for a unresolved cell of the sudoku.
    random.shuffle(open_ass)
    for sl in open_ass:
        #safety check
        if (smart_literal[0][sl] not in assignment) & (-smart_literal[0][sl] not in assignment):
            return smart_literal[0][sl]
    
    next_lit = nextLiteral(cnf)
    print( "No good sugestion, next used instead", next_lit)       
    return next_lit
    
    
    
        

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
            
            
            
            


# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:05:46 2019

@author: Victor Zuanazzi
"""

from heuristics import DLIS, BOHM, randomChoice, nextLiteral, paretoDominant
from abstract_heuristics import learnedHeuristic
from time import time
import numpy as np
import pandas as pd
from os import listdir, remove
from os.path import isfile, join
from load_cnf import parse_cnf
import pickle
import seaborn as sns
import random
from DP import solve, Heuristic
import utils
from load_cnf import load_cnf
from tqdm import tqdm 
from scipy import stats

#tqdm.pandas()

def runExperiments(num_exp = 2, new_exp = True):
    

#    enum_heu = {"Next": 1,
#                "DLIS": 2,
#                "DLIS max": 3,
#                "BOHM": 4,
#                "Pareto Dominant": 5,
#                "Random Forest": 6,
#                "Random": 7}
    enum_heu = {"Random Forest": 6}
    
    h = len(enum_heu)
    
    cols = ["sudoku name", 
            "heuristic", 
            "DP_calls", 
            "backtracks",
            "split_calls", 
            "split_time",
            "assign_calls", 
            "assign_time", 
            "unit_clause_calls",
            "unit_clause_time",
            "solved_sudoku",
            "solve_time"]
    
    df_exp = pd.DataFrame(data = None, columns = cols)
    
    # load rules
    rule_path = '../data/sudoku-rules.txt'
    with open(rule_path, 'r') as file:
            rules = file.read()
    
    # load sudokus
    #path = '../data/dimac_sudoku/'
    path = '../data/hard_sudoku/'
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(onlyfiles)
    
    filename = '..//data//experiment_stats.csv'
    if new_exp:
        df_exp.to_csv(filename, mode = 'w')
    else:
        df_exp.to_csv(filename, mode = 'a', header = False)
    
    for h, heu in enumerate(enum_heu):
        print(f"heuristic: {heu} {h+1}/{len(enum_heu)} ")
        stop_after = num_exp
        for idx, f in enumerate(onlyfiles):
            
            if idx % 9 == 0:
                print(f"{heu} sudoku: {f[20:]} {idx+1}/{num_exp}              "
                      , end = "\n")
                
            #stops once the number of experiments has been reached.
            if idx >= stop_after:
                break
            
            #start counting time
            start = time()
            
            #open file
            with open(f, 'r') as file:
                sudoku = file.read()            

            #rule and sudoku are put together and parsed.
            dimacs = rules + sudoku
            cnf = parse_cnf(dimacs)
            #solves the sudoku and get stats.
            assignment, dict_stats = solve(cnf, Heuristic(h+1))
            
            if dict_stats["split_calls"] == 0:
                #delete trivial sudokus
                print(f"Removed: {f}")
                remove(f)
                continue
            
            dict_stats["heuristic"] = heu
            dict_stats["sudoku name"] = f        
            dict_stats["solve_time"] = (time() - start) #seconds
            
            if(len(assignment) > 0):
                #valid solution
                dict_stats["solved_sudoku"] = 1
                
                # check if number of positive assignments is 81 
                if(len([a for a in assignment if a > 0]) != 81):
                    #invalid solution
                    dict_stats["solved_sudoku"] = 0
                    print("Assignment length incorrect: ", len(assignment))
            else:
                #sudoku is not yet solved
                dict_stats["solved_sudoku"] = 0
                print("Sudoku not solved =/")
            
            if idx % 9 == 0:
                print(f"{heu} sudoku: {f[20:]} {idx+1}/{num_exp}              "
                      , end = "\n")
                print(dict_stats)
                
            #save stats in a dataframe.
            df_exp = df_exp.append(dict_stats, ignore_index = True)
            
            
        #saves the experiments after every heuristic is over.
        repeat = True
        while repeat:
            try:
                df_exp.to_csv(filename, mode = 'a', header = False)
                repeat = False
            except:
                repeat = True
    
    #pickle.dump(df_exp, open(filename, 'wb'))

def statisticalSignificance(df_exp, heuristics, metric = "split_calls", save = True):
    """Computes statistical significance using Kolmogorov–Smirnov 2sample test.
    
    Inputs:
        df_exp: (pd.DataFrame) table containing the metric in a column.
        heuristic: (list(str)) identifier of the rows to be considered, all of 
            them will be evaluated in pair wise fashon.
        metric: (str) name of the column containing the results of interest.
    Output: (pd.Dataframe) a table containing the p-values
    Saves: (.csv) saves the table containing the p-values
    """
    
    #initialize dataframe
    p_value = pd.DataFrame(data = None, columns = heuristics)
    p_value["heuristic"] = None
    p_value.set_index("heuristic")
    
    #dict to store temporary calculations
    pv = {x: 0 for x in p_value.columns}
    
    
    for heu_1 in heuristics:
        
        #select the rows linked to heu_1 
        df_heu1 = df_exp[df_exp["heuristic"] == heu_1][metric]
        pv["heuristic"] = heu_1
        for heu_2 in heuristics:
            
            #select the 
            df_heu2 = df_exp[df_exp["heuristic"] == heu_2][metric]
            _, pv[heu_2] =  stats.ks_2samp(df_heu1, df_heu2)
    
        
        p_value = p_value.append(pv, ignore_index = True)
    
    if save:
        filename = '..//data//p_value_' + metric + '.csv'
        p_value.to_csv(filename)
    
    return p_value

if __name__ == "__main__":
    
    runs = 10
    
    for i in range(runs):
        start = time()        
        runExperiments(num_exp = 1, new_exp= False)
        end = time() - start
        print(f"run {i+1}/{runs}, took: {end/60}min , finishes in: {(runs-i-1)*end/60}min")

#    #load saved experiments   
    filename = '..//data//experiment_stats.csv'
    df_exp = pd.read_csv(filename)
    
    
    heuristics =["random", 
                 "next", 
                 "DLIS", 
                 "DLIS_max", 
                 "BOHM", 
                 "paretoDominant", 
                 "RF"]
    
    y_labels = ["DP_calls", 
                "split_calls",
                "backtracks",
                "unit_clause_calls",
                "solved_sudoku",
                "split_time", 
                "assign_calls",  
                "assign_time",
                "unit_clause_time",         
                "solve_time"]
    
    for l in y_labels:
        statisticalSignificance(df_exp, heuristics, metric = l, save = True)
    
#    df_exp = pickle.load(open(filename, 'rb'))
#    
#    x_categoricals = ["heuristic"]
#                
#    
#    x_numericals = ["DP_calls",
#                    "backtrak"
#                    "split_calls", 
#                    "unit_clause_calls"]
#    

#    
#    hue_labels = [None]
#    
#    fails = plotAllThat(df_exp, x_labels, y_labels, hue_labels)
##    
##    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
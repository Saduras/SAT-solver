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
from os import listdir
from os.path import isfile, join
from load_cnf import parse_cnf
import pickle
import seaborn as sns
import random
from DP import solve
import utils
from load_cnf import load_cnf
from tqdm import tqdm 
#tqdm.pandas()

def runExperiments(num_exp = 2, new_exp = True):
    
    heuristics =["random", 
                 "next", 
                 "DLIS", 
                 "DLIS_max", 
                 "BOHM", 
                 "paretoDominant", 
                 "RF"]
    
    h = len(heuristics)
    
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
    path = '../data/dimac_sudoku/'
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(onlyfiles)
    
    filename = '..//data//experiment_stats.csv'
    if new_exp:
        df_exp.to_csv(filename, mode = 'w')
        
    for h, heu in enumerate(heuristics):
        print(f"heuristic: {heu} {h+1}/{len(heuristics)} ")
        for idx, f in enumerate(onlyfiles):
            
            
            if idx % 9 == 0:
                print(f"{heu} sudoku: {f[20:]} {idx+1}/{num_exp}              "
                      , end = "\n")
                
            #stops once the number of experiments has been reached.
            if idx >= num_exp:
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
            assignment, stats = solve(cnf, heu)
            
            stats["heuristic"] = heu
            stats["sudoku name"] = f        
            stats["solve_time"] = (time() - start) #seconds
            
            if(len(assignment) > 0):
    
                #valid solution
                stats["solved_sudoku"] = 1
                
                # check if number of positive assignments is 81 
                if(len([a for a in assignment if a > 0]) != 81):
                    #invalid solution
                    stats["solved_sudoku"] = 0
                    print("Assignment length incorrect: ", len(assignment))
                    
            else:
                #sudoku is not yet solved
                stats["solved_sudoku"] = 0
                print("Sudoku not solved =/")
                    
            df_exp = df_exp.append(stats, ignore_index = True)
            
            
        #saves the expiriments 
        df_exp.to_csv(filename, mode = 'a', header = False)
                    
    
    #pickle.dump(df_exp, open(filename, 'wb'))
    

if __name__ == "__main__":
    runExperiments(num_exp = 2, new_exp= True)
    
#     #load saved experiments   
#    filename = '..//data//experiment_stats.csv'
    #df_exp = read_csv(filename)
#    df_exp = pickle.load(open(filename, 'rb'))
#    
#    x_categoricals = ["heuristic"]
#                
#    
#    x_numericals = ["DP_calls",
#                    "back"
#                    "split_calls", 
#                    "unit_clause_calls"]
#    
#    y_labels = ["DP_calls", 
#                "split_calls"
#                "backtracks"
#                "unit_clause_calls",
#                "solved_sudoku",
#                "split_time", 
#                "assign_calls",  
#                "assign_time",
#                "unit_clause_time",         
#                "solve_time"]
#    
#    hue_labels = [None]
#    
#    fails = plotAllThat(df_exp, x_labels, y_labels, hue_labels)
##    
##    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
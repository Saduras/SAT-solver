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

def hasEmptyClause(cnf):
    return any([len(c) == 0 for c in cnf])

def removeTautology(cnf):
    for clause in cnf:
        pos_vars = set()
        neg_vars = set()

        for literal in clause:
            if(literal < 0):
                neg_vars.add(abs(literal))
            else:
                pos_vars.add(abs(literal))

        for literal in clause:
            if(abs(literal) in pos_vars and abs(literal) in neg_vars):
                cnf.remove(clause)
                break

    return cnf

def assign(literal, value, cnf, assignment, stats, heuristic = "next"):

    stats["assign_calls"] += 1
    startTime = time()

    # Add to assignment
    result_assignment = assignment.copy()
    assign = literal
    if(not value):
        assign *= -1
    result_assignment.append(assign)

    result_cnf = [clause.copy() for clause in cnf]

    # Remove literals from cnf
    for clause in result_cnf.copy():
        if((literal in clause and value) 
        or (-literal in clause and not value)):
            # literal becomes true; clause disappears
            result_cnf.remove(clause)
        else:
            # literal becomes false; literal is removed
            clause.pop(literal, None)
            clause.pop(-literal, None)

    stats["assign_time"] += time() - startTime
    
    return result_cnf, heuristic,  stats, result_assignment

def removeUnitClause(cnf, assignment, stats):
    
    stats["unit_clause_calls"] += 1
    startTime = time()

    change = False
    loop = True
    
    while loop:
        # stop looping if nothing changes this iteration
        loop = False
        for clause in cnf:
            if len(clause) == 1:
                # unit clause found; continue looping
                loop = True
                change = True

                cnf, _, stats, assignment = assign(list(clause.keys())[0], 
                                                True,
                                                cnf, 
                                                assignment,
                                                stats)
                break

    stats["unit_clause_time"] += time() - startTime
    
    return cnf, assignment, change, stats

def removePureLiteral(cnf, assignment):
    # TODO: consider removing or optimizing
    return cnf, assignment, False

def choseLiteral(cnf, assignment, choice = "next"):
    """choose literal and it's assigned value based on heuristics
    input: 
        cnf: the cnf in the standard format
        choice: the name of the heuristic to be executed.
            "random", "DLIS", "BOHM", "next"
    output: 
        literal key,
        value for the assigment.
    """
    
    #naive implementation:
    if choice == "random":
        return randomChoice(cnf) 
    elif choice == "DLIS": # worst choice for split and time
        return DLIS(cnf, take = "min") 
    elif choice == "DLIS_max":
        return DLIS(cnf, take = "max") 
    elif choice == "BOHM": #best choice for split
        return BOHM(cnf)
    elif choice == "paretoDominant":
        return paretoDominant(cnf)
    elif choice == "RF":
        return learnedHeuristic(cnf, assignment)
    else:
        return nextLiteral(cnf) #best choice for time
    
def split(value, cnf, assignment, heuristic, stats):
    
    stats["split_calls"] += 1
    startTime = time()

    if(len(cnf) == 0 or len(cnf[0]) == 0):
        raise Exception("Invalid CNF to split on! CNF or 1st clause are empty!", cnf)

    literal, _ = choseLiteral(cnf, assignment, choice = heuristic)
    
    stats["split_time"] = time() - startTime
                  
    return assign(literal, value, cnf, assignment, stats, heuristic)

def DP(cnf, heuristic, stats, assignment = []):
    """
    Solves a SAT-problem given in clausal normal form (CNF) using the Davis-
        Putnam algorithm
    input:
        cnf - SAT-problem in clausal normal form
        assignment - list of already assigned truth values; leave empty
    output:
        If satisfiable: list of assignments for solution
        else: empty list
    """
        
    stats["DP_calls"] += 1

    # success condition: empty set of clauses
    if(len(cnf) == 0):
        return assignment, stats

    # failure condition: empty clause
    if(hasEmptyClause(cnf)):
        return [], stats
    
    # stuck sudoku
    if stats["DP_calls"] > 100:
        print("You suck!")
        return assignment, stats
    
    # simplification
    cnf, assignment, done, stats = removeUnitClause(cnf, assignment, stats)
    
    if(not done):
        cnf, assignment, done = removePureLiteral(cnf, assignment)
        
    if(done):
        return DP(cnf, heuristic, stats, assignment)
    
    else:
        solved_assignment, stats = DP(*split(True,
                                             cnf,
                                             assignment,
                                             heuristic,
                                             stats))
        
        # split with True satisfied
        if(len(solved_assignment) != 0):
            return solved_assignment, stats
        
        # or didn't work; then try False
        else:
            #Why does it call split() instead of assign()?
            stats["backtracks"] += 1
            return DP(*split(False, 
                             cnf, 
                             assignment, 
                             heuristic, 
                             stats))
        
    
def solve(cnf, heuristic, log = False):
    
    stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "backtracks": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }
    
    #removes Tautologies
    cnf = removeTautology(cnf)
    
    #Davis Putnam Sat Sover
    assignment, stats = DP(cnf, heuristic, stats)

    if(log):
        print("Satisfiable:", len(assignment) > 0)
        print(f"DP calls {stats['DP_calls']}")
        if(stats["assign_calls"] > 0):
            print(f"assign calls: {stats['assign_calls']} total time: {stats['assign_time']:.2f}s avg time: {stats['assign_time']/stats['assign_calls'] * 1000:.3f}ms")
        if(stats["unit_clause_calls"] > 0):
            print(f"unitClause calls: {stats['unit_clause_calls']} total time: {stats['unit_clause_time']:.2f}s avg time: {stats['unit_clause_time']/stats['unit_clause_calls'] * 1000:.3f}ms")
        if(stats["split_calls"] > 0):
            print(f"split calls: {stats['split_calls']} total time: {stats['split_time']:.2f}s avg time: {stats['split_time']/stats['split_calls'] * 1000:.3f}ms")
            print(f"backtrcks: {stats['backtracks']}")
    return assignment, stats
    

def alternative_main():
    from load_cnf import load_cnf
    
    filename = ".\\data\\sudoku-example-processed.txt"
    cnf = load_cnf(filename)
    assignment = solve(cnf)
    
    
    #print sudoku in a (almost) human readible way
    sudoku_solution = [a for a in assignment if a > 0]
    #print(sudoku_solution)
    print(len(sudoku_solution))
    matrix = [ [ 0 for i in range(9) ] for j in range(9) ]
    for value in sudoku_solution:
        v = [int(i) for i in str(value)]
        #print(v)
        matrix[v[0] - 1][v[1] - 1] = v[2]
    print(matrix)
    
    
def titleNAxis(fig_plot, title, label_x, label_y):
    """enter title and axis labels with standard font sizes.
    """
    fig_plot.set_title(title, fontsize = 40)
    fig_plot.set_xlabel(label_x, fontsize = 30)
    fig_plot.set_ylabel(label_y, fontsize = 30)
    
    return fig_plot
    
def savePlot(fig_plot, fig_name, path = ".//figures//"):
    """saves a seaborn figure
    """
    try:
        fig = fig_plot.get_figure()
        file_path = path + fig_name + ".png"
        fig.savefig(file_path)
        return file_path
    except:
        return False
    
def plotAllThat(df_exp, x_labels, y_labels, hue_labels, path = ".//figures//", plot_type = "all"):
    
    size_x = 30
    size_y = 15
    
    sns.set(rc={'figure.figsize':(size_x,size_y)}) #in inches
    fails ={"barplot": [0, []],
            "boxplot": [0, []],
            "swarmplot": [0, []],
            "catplot": [0, []]}
    
    for i, label_i in enumerate(x_labels):
        for j, label_j in enumerate(y_labels):
            for k, label_k in enumerate(hue_labels):
                title = label_i + " vs " + label_j + " " + str(label_k)
                
                #barplot
                if (plot_type == "barplot") | (plot_type == "all"):       
                    try:
                        f_plot = sns.barplot(x = label_i, y = label_j, hue = label_k, data = df_exp)   
                        f_plot = titleNAxis(f_plot, title, label_i, label_j)
                        file_path = savePlot(f_plot, title + " barplot")
                    except:
                        fails["barplot"][0] += 1
                        fails["barplot"][1].append((label_i, label_j))
   
# FIND OUT WHY THOSE PLOTS WONT WORK                  
#                #boxplot
#                if (plot_type == "boxplot") | (plot_type == "all"):
#                    try:
#                        f_plot = sns.boxplot(x = label_i, y = label_j, hue = label_k, data = df_exp)   
#                        f_plot = titleNAxis(f_plot, title, label_i, label_j)
#                        f_plot.set_size_inches(size_x, size_y)
#                        file_path = savePlot(f_plot, title + " boxolot")
#                    except:
#                         fails["boxplot"][0] += 1
#                         fails["boxplot"][1].append((label_i, label_j))
#                         
#                #swarmplot
#                if (plot_type == "swarmplot") | (plot_type == "all"):
#                    try:
#                        f_plot = sns.swarmplot(x = label_i, y = label_j, hue = label_k, data = df_exp)   
#                        #f_plot.set_size_inches(size_x, size_y)
#                        f_plot = titleNAxis(f_plot, title, label_i, label_j)
#                        file_path = savePlot(f_plot, title + " swarmplot")
#                    except:
#                        fails["swarmplot"][0] += 1
#                        fails["swarmplot"][1].append((label_i, label_j))
                        
#                if (plot_type == "catplot") | (plot_type == "all"):
#                    try:
#                        f_plot = sns.catplot(x = label_i, 
#                                             y = label_j, 
#                                             hue = label_k, 
#                                             data = df_exp, 
#                                             height= size_y,
#                                             aspect=size_x/size_y)   
#                        file_path = savePlot(f_plot, title + " catplot")
#                    except:
#                        fails["catplot"][0] += 1
#                        fails["catplot"][1].append((label_i, label_j))
        
    return fails

def runExperiments(num_exp = 2):
    
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
            "backtracks"
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
    rule_path = './data/sudoku-rules.txt'
    with open(rule_path, 'r') as file:
            rules = file.read()
    
    # load sudokus
    path = './data/dimac_sudoku/'
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(onlyfiles)
    
    for h, heu in enumerate(heuristics):
        print(heu)
        for idx, f in enumerate(onlyfiles):
            
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
            assignment, stats = solve(cnf, heu, log = True)
            
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
                    print("NOT Solved")
                    
            df_exp = df_exp.append(stats, ignore_index = True)
                    
    #saves the expiriments
    filename = 'experiment_stats.sav'
    pickle.dump(df_exp, open(filename, 'wb'))
    


if __name__ == "__main__":
    runExperiments(num_exp = 1)
    
#     #load saved experiments   
    filename = 'experiment_stats.sav'
    df_exp = pickle.load(open(filename, 'rb'))
    
    x_labels = ["heuristic", 
                "DP_calls" 
                "split_calls", 
                "unit_clause_calls"]
    
    y_labels = ["DP_calls", 
                "split_calls"
                "backtracks"
                "unit_clause_calls",
                "solved_sudoku",
                "split_time", 
                "assign_calls",  
                "assign_time",
                "unit_clause_time",         
                "solve_time"]
    
    hue_labels = [None]
    
    fails = plotAllThat(df_exp, x_labels, y_labels, hue_labels)
#    
#    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
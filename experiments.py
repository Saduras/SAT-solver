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

DEBUG = True
SAVE_SPLIT = True
DPcalls = 0
splitCalls = 0
splitTime = 0
assignCalls = 0
assignTime = 0
unitClauseCalls = 0
unitClauseTime = 0

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

def assign(literal, value, cnf, assignment, stats):

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
    
    return result_cnf, result_assignment, stats

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

                cnf, assignment, stats = assign(list(clause.keys())[0], 
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
        return learnedHeuristic(assignment)
    else:
        return nextLiteral(cnf) #best choice for time
    
def split(value, cnf, assignment, heuristic, stats):
    
    stats["split_calls"] += 1
    startTime = time()

    if(len(cnf) == 0 or len(cnf[0]) == 0):
        raise Exception("Invalid CNF to split on! CNF or 1st clause are empty!", cnf)
        
    # take 1st literal
    literal, _ = choseLiteral(cnf, assignment, choice = heuristic)
    
    stats["split_time"] = time() - startTime
        
    return assign(literal, value, cnf, assignment, stats)

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
            return DP(*split(False, 
                             cnf, 
                             assignment, 
                             heuristic, 
                             stats))
        
    
def solve(cnf, heuristic, log = False):
    
    stats = {
            "DP_calls": 0,
            "split_calls": 0,
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
        if(assignCalls > 0):
            print(f"assign calls: {stats['assign_calls']} total time: {stats['assign_time']:.2f}s avg time: {stats['assign_time']/stats['assign_calls'] * 1000:.3f}ms")
        if(unitClauseCalls > 0):
            print(f"unitClause calls: {stats['unit_clause_calls']} total time: {stats['unit_clause_time']:.2f}s avg time: {stats['unit_clause_time']/stats['unit_clause_calls'] * 1000:.3f}ms")
        if(splitCalls > 0):
            print(f"split calls: {stats['split_calls']} total time: {stats['split_time']:.2f}s avg time: {stats['splitTime']/stats['splitCalls'] * 1000:.3f}ms")
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
    
def runExperiments(num_exp = 1):
    
    heuristics =["random", 
                 "next", 
                 "DLIS", 
                 "DLIS_max", 
                 "BOHM", 
                 "paretoDominant", 
                 "RF"]
    
    h = len(heuristics)
    
    DP_calls = np.zeros((h, num_exp))
    split_calls = np.zeros((h, num_exp))
    split_time = np.zeros((h, num_exp))
    assign_calls = np.zeros((h, num_exp))
    assign_time = np.zeros((h, num_exp))
    unit_clause_calls = np.zeros((h, num_exp))
    unit_clause_time = np.zeros((h, num_exp))
    solved_sudoku = np.zeros((h, num_exp))
    solve_time = np.zeros((h, num_exp))
    
    # load rules
    rule_path = './data/sudoku-rules.txt'
    with open(rule_path, 'r') as file:
            rules = file.read()
    
    # load sudokus
    path = './data/dimac_sudoku/'
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    
    for h, heu in enumerate(heuristics):
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
            
            #store stats.
            DP_calls[h][idx] = stats["DP_calls"]
            split_calls[h][idx] = stats["split_calls"]
            split_time[h][idx] = stats["split_time"]
            assign_calls[h][idx] = stats["assign_calls"]
            assign_time[h][idx] = stats["assign_time"]
            unit_clause_calls[h][idx] = stats["unit_clause_calls"]
            unit_clause_time[h][idx] = stats["unit_clause_time"]
            solve_time[h][idx] = (time() - start) #seconds
            
            if(len(assignment) > 0):
                #valid solution
                solved_sudoku[h][idx] = 1
                # check if number of positive assignments is 81 
                if(len([a for a in assignment if a > 0]) != 81):
                    #invalid solution
                    solved_sudoku[h][idx] = 2
                    print("NOT Solved")
                    print(assignment)
                    
                    
    print(f"DP_calls: \n{DP_calls}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"split_calls: \n{split_calls}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"split_time: \n{split_time}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"assign_calls: \n{assign_calls}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"assign_time: \n{assign_time}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"unit_clause_calls: \n{unit_clause_calls}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"unit_clause_time: \n{unit_clause_time}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"solve_time: \n{solve_time}")
    print("^^^^^^^^^^^^^^^^^^^^^^")
    print(f"solved_sudoku: \n{solved_sudoku}")
        
                    
         #load saved model.    
    filename = 'experiment_stats.sav'
    pickle.dump(DP_calls, open(filename, 'wb'))
    


if __name__ == "__main__":
    runExperiments()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
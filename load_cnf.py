# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:44:05 2019

@author: Victor Zuanazzi
"""

def load_cnf(path):
    """load a txt file into a DIMAC format.
    input:
        path: path to the file
    return: list of list of tuples
        [[(clause (type int), negation (type boolean))]]
    """
    
    #load file
    with open(path, 'r') as f:
        lines = f.readlines()
    
    #cnf is stored in a list.
    cnf = []
    
    for l in lines:
        
        #excludes lines wihout clauses
        if "c" == l[0] or "p" == l[0]:
            continue
        
        #extracts all the clauses from one line
        clause_split = l.split(" ")[:-1] #the 0 in the end of the line is ignored.
        clause = [] 
        for c in clause_split:
            #clause is loaded as a tuple of int and boolean
            clause.append( (abs(int(c)), c[0] == '-') )
        
        cnf.append(clause)

    return cnf

if __name__ == "__main__":
    cnf = load_cnf(".\data\sudoku-example-processed.txt")


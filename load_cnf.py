# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:44:05 2019

@author: Victor Zuanazzi
"""

def load_cnf(path):
    """load a txt file into a DIMAC format.
    input:
        path: path to the file
    return: list of default dictionaries
        [{literal: True},{otherLiteral: True}]
    """
    with open(path, 'r') as f:
        dimacs = f.read()

    return parse_cnf(dimacs)

def parse_cnf(dimacs):
    """parse a string in DIMAC format
    input:
        dimacs: string in DIMAC format
    return: list of default dictionaries
        [{literal: True},{otherLiteral: True}]
    """
    lines = dimacs.split('\n')
    
    #cnf is stored in a list.
    cnf = []
    
    for l in lines:
        #excludes lines wihout clauses
        if len(l) == 0 or "c" == l[0] or "p" == l[0]:
            continue
        
        #extracts all the clauses from one line
        clause_split = l.split(" ")[:-1] #the 0 in the end of the line is ignored.
        clause = {}
        for literal in clause_split:
            clause[int(literal)] = True
        
        cnf.append(clause)

    return cnf

if __name__ == "__main__":
    cnf = load_cnf(".\data\sudoku-example-processed.txt")


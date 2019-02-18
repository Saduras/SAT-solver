from heuristics import DLIS, BOHM, randomChoice, nextLiteral
from time import time
import numpy as np
import pandas as pd
import csv


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

def assign(literal, value, cnf, assignment):
    if(DEBUG):
        global assignCalls, assignTime
        assignCalls += 1
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

    if(DEBUG):
        assignTime += time() - startTime
    return result_cnf, result_assignment

def removeUnitClause(cnf, assignment):
    if(DEBUG):
        global unitClauseCalls, unitClauseTime
        unitClauseCalls += 1
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

                cnf, assignment = assign(list(clause.keys())[0], True, cnf, assignment)
                break

    if(DEBUG):
        unitClauseTime += time() - startTime
    return cnf, assignment, change

def removePureLiteral(cnf, assignment):
    # TODO: consider removing or optimizing
    return cnf, assignment, False

def choseLiteral(cnf, choice = "next"):
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
        return DLIS(cnf) 
    elif choice == "BOHM": #best choice for split
        return BOHM(cnf)
    else:
        return nextLiteral(cnf) #best choice for time
     
    #efficient implementation (not working)
#    heuristics = {
#            "DLIS": DLIS(cnf),
#            "BOHM": BOHM(cnf),
#            "random": randomChoice(cnf),
#            "next": nextLiteral(cnf)
#            }
    
#    return heuristics.get(choice, nextLiteral(cnf))
    
def saveSplit(cnf):
    """save all splits in a .csv
    """
    
    
    #enough space for all the clauses and literals
    clauses = np.zeros((12006,10))
    
    df = pd.DataFrame(columns = ["clauses"])
    df = df.append(pd.Series(data = {"clauses": [None]}),
                   ignore_index = True)
    
    #vector implementation if this ugly thing?
    for c, clause in enumerate(cnf):
        for l, literal in enumerate(clause):
           clauses[c][l] = literal
         
    #df = pd.DataFrame(zip(clauses), columns = ["clauses"])
    df.loc[0, "clauses"] = [int(clauses[i][j]) for i in range(len(clauses)) for j in range(len(clauses[i])) ] #np.reshape(clauses, (1,12006*10))
    
    path = './data/Splits.csv'
    try:
        df.to_csv(path_or_buf = path, mode = 'a', header = False)
    except:
        pass

    
def split(value, cnf, assignment):
    if(DEBUG):
        global splitCalls, splitTime
        splitCalls += 1
        startTime = time()

    if(len(cnf) == 0 or len(cnf[0]) == 0):
        raise Exception("Invalid CNF to split on! CNF or 1st clause are empty!", cnf)
        
    # take 1st literal
    literal, _ = choseLiteral(cnf, "next")
    
    if SAVE_SPLIT:
        saveSplit(cnf)   
        
    if(DEBUG):
        splitTime = time() - startTime
    return assign(literal, value, cnf, assignment)

def DP(cnf, assignment = []):
    """
    Solves a SAT-problem given in clausal normal form (CNF) using the Davis-Putnam algorithm
    input:
        cnf - SAT-problem in clausal normal form
        assignment - list of already assigned truth values; leave empty
    output:
        If satisfiable: list of assignments for solution
        else: empty list
    """
    if(DEBUG):
        global DPcalls
        DPcalls += 1

    # success condition: empty set of clauses
    if(len(cnf) == 0):
        return assignment

    # failure condition: empty clause
    if(hasEmptyClause(cnf)):
        return []
    
    # simplification
    cnf, assignment, done = removeUnitClause(cnf, assignment)
    if(not done):
        cnf, assignment, done = removePureLiteral(cnf, assignment)
    if(done):
        return DP(cnf, assignment)
    else:
        solved_assignment = DP(*split(True, cnf, assignment))
        # split with True satisfied
        if(len(solved_assignment) != 0):
            return solved_assignment
        # or didn't work; then try False
        else:
            return DP(*split(False, cnf, assignment))
        
def saveLabel(assignment, n_splits):
    """save the finished sudoku in a .csv
    """
    
    sudoku_solution = [a for a in assignment if a > 0]
    
    sudoku_solution.sort()
     
    label = np.reshape(np.array(sudoku_solution*n_splits), 
                       (n_splits, len(sudoku_solution)))

    df = pd.DataFrame(zip(label))
    
    path = './data/SplitsLabel.csv'
    try:
        df.to_csv(path_or_buf = path, 
                  mode = 'a',
                  header = False)
    except:
        pass
    
def solve(cnf):
    if(DEBUG):
        global DPcalls, assignCalls, assignTime, unitClauseCalls, unitClauseTime, splitCalls, splitTime
        DPcalls = 0
        splitCalls = 0
        splitTime = 0
        assignCalls = 0
        assignTime = 0
        unitClauseCalls = 0
        unitClauseTime = 0

    cnf = removeTautology(cnf)
    assignment = DP(cnf)
    
    if SAVE_SPLIT:
        saveLabel(assignment, splitCalls)

    if(DEBUG):
        print("Satisfiable:", len(assignment) > 0)
        print(f"DP calls {DPcalls}")
        if(assignCalls > 0):
            print(f"assign calls: {assignCalls} total time: {assignTime:.2f}s avg time: {assignTime/assignCalls * 1000:.3f}ms")
        if(unitClauseCalls > 0):
            print(f"unitClause calls: {unitClauseCalls} total time: {unitClauseTime:.2f}s avg time: {unitClauseTime/unitClauseCalls * 1000:.3f}ms")
        if(splitCalls > 0):
            print(f"split calls: {splitCalls} total time: {splitTime:.2f}s avg time: {splitTime/splitCalls * 1000:.3f}ms")
    return assignment

def main():
    from load_cnf import load_cnf
    import sys

    if(len(sys.argv) != 2):
        print("Missing arguments! Expecting 'DIMACS file path'")
        exit()

    filename = sys.argv[1]

    cnf = load_cnf(filename)
    assignment = solve(cnf)
    
    sudoku_solution = [a for a in assignment if a > 0]
    print(sudoku_solution)
    print(len(sudoku_solution))
    matrix = [ [ 0 for i in range(9) ] for j in range(9) ]
    for value in sudoku_solution:
        v = [int(i) for i in str(value)]
        print(v)
        matrix[v[0] - 1][v[1] - 1] = v[2]
    print(matrix)
    


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
    

if __name__ == "__main__":
    alternative_main()
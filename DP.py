from heuristics import DLIS, BOHM, randomChoice, nextLiteral, paretoDominant
from abstract_heuristics import learnedHeuristic
from time import time
import warnings

MAX_RECURSIONS = 100

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

def assign(literal, value, cnf, assignment, stats = None):

    if(stats):
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

    if(stats):
        stats["assign_time"] += time() - startTime
    
    return result_cnf, result_assignment, stats

def removeUnitClause(cnf, assignment, stats = None):
    if(stats):
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

    if(stats):
        stats["unit_clause_time"] += time() - startTime
    
    return cnf, assignment, change, stats

def choseLiteral(cnf, assignment, choice = "next"):
    """choose literal and it's assigned value based on heuristics
    input: 
        cnf: the cnf in the standard format
        choice: the name of the heuristic to be executed.
            "random", "DLIS", "BOHM", "next"
    output: 
        literal key,
        value for the assignment.
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
 
def split(cnf, assignment, heuristic, stats = None):
    
    if(stats):
        stats["split_calls"] += 1
        startTime = time()

    if(len(cnf) == 0 or len(cnf[0]) == 0):
        raise Exception("Invalid CNF to split on! CNF or 1st clause are empty!", cnf)

    literal, _ = choseLiteral(cnf, assignment, choice = heuristic)
    
    if(stats):
        stats["split_time"] = time() - startTime
    
    return literal, stats

def DP(cnf, heuristic=None, onSplit=None, stats=None, assignment = []):
    """
    Solves a SAT-problem given in clausal normal form (CNF) using the Davis-
        Putnam algorithm
    input:
        cnf - SAT-problem in clausal normal form
        heuristic - which heuristic to use to decide on splits
        onSplit - call back just before every split, arguments: (cnf, assignment)
        stats - dictionary with runtime stats; leave empty
        assignment - list of already assigned truth values; leave empty
    output:
        assignment - if satisfiable: list of assignments for solution 
            else empty list
        stats - dictionary with stats about run
    """
    
    if(stats):
        stats["DP_calls"] += 1

    # success condition: empty set of clauses
    if(len(cnf) == 0):
        return assignment, stats

    # failure condition: empty clause
    if(hasEmptyClause(cnf)):
        return [], stats
    
    # stuck SAT problems
    if stats and stats["DP_calls"] > MAX_RECURSIONS:
        warnings.warn(f"Reached limit of {MAX_RECURSIONS} recursions.", RuntimeWarning)
        return assignment, stats
    
    # simplification
    cnf, assignment, done, stats = removeUnitClause(cnf, assignment, stats)
    if(done):
        return DP(cnf, heuristic, onSplit, stats, assignment)
    
    else:
        if onSplit:
            onSplit(cnf, assignment)  
        
        literal, stats = split(cnf, assignment, heuristic, stats)
        new_cnf, new_assignment, stats = assign(literal, True, cnf, assignment, stats)
        solved_assignment, stats = DP(new_cnf, heuristic, onSplit, stats, new_assignment)
        
        # split with True satisfied
        if(len(solved_assignment) != 0):
            return solved_assignment, stats
        
        # or didn't work; then try False
        else:
            #Why does it call split() instead of assign()?
            if(stats):
                stats["backtracks"] += 1
            
            literal, stats = split(cnf, assignment, heuristic, stats)
            new_cnf, new_assignment, stats = assign(literal, False, cnf, assignment, stats)
            return DP(new_cnf, heuristic, onSplit, stats, new_assignment)
    
def solve(cnf, heuristic=None, onSplit=None):
    """
    Solves given SAT problem with Davis Putnam after removing tautology clauses
    input:
        cnf - SAT problem in clausal normal form; expects clauses to be 
            dictionaries of literals
        heuristic - which heuristic to use to decide on splits
        onSplit - call back just before every split, arguments: (cnf, assignment)
    output:
        assignment - if satisfiable: list of assignments for solution 
            else empty list
        stats - dictionary with stats about run
    """
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
    
    cnf = removeTautology(cnf)
    
    #Davis Putnam Sat Solver
    return DP(cnf, heuristic, onSplit, stats = stats)

def print_stats(assignment, stats):
    print("Satisfiable:", len(assignment) > 0)
    print(f"DP calls {stats['DP_calls']}")
    if(stats["assign_calls"] > 0):
        print(f"assign calls: {stats['assign_calls']} total time: {stats['assign_time']:.2f}s avg time: {stats['assign_time']/stats['assign_calls'] * 1000:.3f}ms")
    if(stats["unit_clause_calls"] > 0):
        print(f"unitClause calls: {stats['unit_clause_calls']} total time: {stats['unit_clause_time']:.2f}s avg time: {stats['unit_clause_time']/stats['unit_clause_calls'] * 1000:.3f}ms")
    if(stats["split_calls"] > 0):
        print(f"split calls: {stats['split_calls']} total time: {stats['split_time']:.2f}s avg time: {stats['split_time']/stats['split_calls'] * 1000:.3f}ms")
        print(f"backtracks: {stats['backtracks']}")

def main():
    from load_cnf import load_cnf
    import sys

    if(len(sys.argv) != 2):
        print("Missing arguments! Expecting 'DIMACS file path'")
        exit()

    filename = sys.argv[1]

    cnf = load_cnf(filename)
    assignment, stats = solve(cnf)
    print_stats(assignment, stats)

if __name__ == "__main__":
    main()
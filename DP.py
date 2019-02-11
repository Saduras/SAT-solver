def hasEmptyClause(cnf):
    return any([len(c) == 0 for c in cnf])

def removeTautology(cnf):
    # TODO
    return cnf

def removeUnitClause(cnf, assignment):
    # TODO
    return cnf, assignment, False

def removePureLiteral(cnf, assignment):
    # TODO: consider removing or optimising
    return cnf, assignment, False

def split(cnf, assignment):
    # TODO
    return cnf, assignment

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
    if(not done):
        cnf, assignment = split(cnf, assignment)
    
    DP(cnf, assignment)

def solve(cnf):
    cnf = removeTautology(cnf)
    return DP(cnf)
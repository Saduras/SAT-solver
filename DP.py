def hasEmptyClause(cnf):
    return any([len(c) == 0 for c in cnf])

def removeTautology(cnf):
    for clause in cnf:
        pos_vars = set()
        neg_vars = set()

        for literal in clause:
            if(literal[1]):
                neg_vars.add(literal[0])
            else:
                pos_vars.add(literal[0])

        for literal in clause:
            if(literal[0] in pos_vars and literal[0] in neg_vars):
                cnf.remove(clause)
                break

    return cnf

def assign(literal, value, cnf, assignment):
    # Add to assignment
    assign = literal[0]
    if(literal[1]):
        assign *= -1
    if(not value):
        assign *= -1
    assignment.append(assign)

    # Remove literals from cnf
    for clause in cnf.copy():
        for otherliteral in clause.copy():
            if otherliteral[0] == literal[0]:
                if((otherliteral[1] == literal[1]) == value):
                    # literal becomes true; clause disappears
                    cnf.remove(clause)
                    break
                else:
                    # literal becomes false; literal is removed
                    clause.remove(otherliteral)

    return cnf, assignment

def removeUnitClause(cnf, assignment):
    change = False
    loop = True
    while loop:
        # stop looping if nothing changes this iteration
        loop = False
        for clause in cnf.copy():
            if len(clause) == 1:
                # unit clause found; continue looping
                loop = True
                change = True

                cnf, assignment = assign(clause[0], True, cnf, assignment)
            
    return cnf, assignment, change

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
import copy

DPcalls = 0
splits = 0

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
    # Add to assignment
    result_assignment = assignment.copy()
    assign = literal
    if(not value):
        assign *= -1
    result_assignment.append(assign)

    result_cnf = copy.deepcopy(cnf)

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

    return result_cnf, result_assignment

def removeUnitClause(cnf, assignment):
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
            
    return cnf, assignment, change

def removePureLiteral(cnf, assignment):
    # TODO: consider removing or optimising
    return cnf, assignment, False

def split(value, cnf, assignment):
    global splits
    splits += 1
    if(len(cnf) == 0 or len(cnf[0]) == 0):
        raise Exception("Invalid CNF to split on! CNF or 1st clause are empty!", cnf)

    # take 1st literal
    literal = list(cnf[0].keys())[0]
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

def solve(cnf):
    global DPcalls, splits
    DPcalls = 0
    splits = 0

    cnf = removeTautology(cnf)
    assignment = DP(cnf)

    # print("Satisfiable:", len(assignment) > 0,"DP calls:", DPcalls, "splits:", splits)
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



if __name__ == "__main__":
    main()
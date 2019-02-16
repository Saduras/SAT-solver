from os import listdir
from os.path import isfile, join
from load_cnf import parse_cnf
from DP import solve

sat = 0
sat_error = 0
unsat = 0

# load rules
rule_path = './data/sudoku-rules.txt'
with open(rule_path, 'r') as file:
        rules = file.read()

# load sudokus
path = './data/dimac_sudoku/'
onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
for idx,f in enumerate(onlyfiles):
    print("Solving",f,idx+1,"/",len(onlyfiles),"...")
    with open(f, 'r') as file:
        sudoku = file.read()

    dimacs = rules + sudoku
    cnf = parse_cnf(dimacs)

    assignment = solve(cnf)

    if(len(assignment) > 0):
        sat += 1
        if(len(assignment) != 81):
            sat_error +=1
    else:
        unsat += 1

print("Results -  sat:", sat, "sat_error:", sat_error, "unsat:", unsat)
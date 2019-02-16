import sys
from load_cnf import parse_cnf
from DP import solve

# load rules
rule_path = './data/sudoku-rules.txt'
with open(rule_path, 'r') as file:
        rules = file.read()

# load sudokus
path = sys.argv[1]
with open(path, 'r') as file:
    sudoku = file.read()

dimacs = rules + sudoku
cnf = parse_cnf(dimacs)

assignment = solve(cnf)
print(assignment)
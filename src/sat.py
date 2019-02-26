import sys
from load_cnf import load_cnf
from DP import solve

if(len(sys.argv) != 2):
    print("Missing arguments! Expecting 'DIMACS file path'")
    exit()

filename = sys.argv[1]

cnf = load_cnf(filename)
assignment, stats = solve(cnf)
print(assignment)
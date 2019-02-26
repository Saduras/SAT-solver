import sys
from load_cnf import load_cnf
from DP import solve, print_stats, Heuristic
from utils import assignmentToDIMACSFile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-S', '--strategy', dest='strategy', type=int, default=1,
                    help='1 for the basic DP and n = 2 or 3 for your two other strategies')
parser.add_argument('filename', help='the input file is the concatenation of all required input clauses.')

args = parser.parse_args()

cnf = load_cnf(args.filename)
assignment, stats = solve(cnf, Heuristic(args.strategy))
print_stats(assignment, stats)

assignmentToDIMACSFile(assignment, f'{args.filename}.out')
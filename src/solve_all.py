from os import listdir
from os.path import isfile, join
from load_cnf import parse_cnf
from DP import solve
from time import time
import numpy as np

sat = 0
sat_error = 0
unsat = 0
save = True
start_at = np.random.randint(17011) #11502 -> 5904
stop_at = min(start_at + 5500, 22011)

# load rules
rule_path = './data/sudoku-rules.txt'
with open(rule_path, 'r') as file:
        rules = file.read()

# load sudokus
path = './data/dimac_sudoku/'
onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
for idx,f in enumerate(onlyfiles):
    
    if idx < start_at:
        continue
    
    start = time()
    with open(f, 'r') as file:
        sudoku = file.read()

    dimacs = rules + sudoku
    cnf = parse_cnf(dimacs)

    assignment, _ = solve(cnf)

    if(len(assignment) > 0):
        sat += 1
        # check if number of positive assignments is 81 (one per sudoku field)
        if(len([a for a in assignment if a > 0]) != 81):
            sat_error +=1
    else:
        unsat += 1
    print("Solved",f,idx+1,"/",len(onlyfiles)," - sat:", sat, "sat_error:", sat_error, "unsat:", unsat, "time:", (time()-start) * 1000,"ms")
    
    #save the finished sudoku
    if save:
        save_path = './data/solved_sudoku/'
        s_name = f.split("/")[-1] + "_solved.txt"
        with open(save_path + s_name, "w+") as file:
            file.writelines("%s\n" % x for x in assignment)
            
    if stop_at and (idx >= stop_at):
        break
    
print("Results - sat:", sat, "sat_error:", sat_error, "unsat:", unsat)
import numpy as np 
import pandas as pd


def saveSS(cnf, assignment):
    """Saves the sudoku state in the file SSplits.csv
    """
    
    #maps the assignment to a position in the sudoku vector.
    #It over-generates, but that is fine for now.
    ass2sud = {x + (i+1)*10: i for i in range(81) for x in range(101, 190) }
    
    sudoku_state = np.zeros((1, 81))
    
    #is there a pythonic way of writing it?
    for a in assignment:
        if a > 0:
            sudoku_state[0][ass2sud[a]] = a
     
    df = pd.DataFrame(data = sudoku_state, 
                      columns = [x for x in range(81)], 
                      dtype = "int_")
    
    path = './data/SSplits.csv'
    try:
        df.to_csv(path_or_buf = path, mode = 'a', header = False)
    except:
        pass
        
def saveLabel(assignment, n_splits):
    """save the finished sudoku in a .csv
    """
    
    sudoku_solution = [a for a in assignment if a > 0]
    
    sudoku_solution.sort()
     
    label = np.reshape(np.array(sudoku_solution*n_splits), 
                       (n_splits, len(sudoku_solution)))

    df = pd.DataFrame(label, columns = [x for x in range(len(sudoku_solution))])
    
    path = './data/SplitsLabel.csv'
    try:
        df.to_csv(path_or_buf = path, 
                  mode = 'a',
                  header = False)
    except:
        pass

def DIMACS_to_sudoku(assignment):
    sudoku_solution = [a for a in assignment if a > 0]
    matrix = [ [ 0 for i in range(9) ] for j in range(9) ]
    for value in sudoku_solution:
        v = [int(i) for i in str(value)]
        matrix[v[0] - 1][v[1] - 1] = v[2]
    return matrix
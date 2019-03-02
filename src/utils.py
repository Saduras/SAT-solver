import numpy as np 
from abstract_heuristics import loadData
import pandas as pd


def saveSS(cnf, assignment, path = '../data/SSplits.csv' ):
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
    
    try:
        df.to_csv(path_or_buf = path, mode = 'a', header = False)
    except:
        pass
        
def saveLabel(assignment, n_splits, path = '../data/SplitsLabel.csv'):
    """save the finished sudoku in a .csv
    """
    
    sudoku_solution = [a for a in assignment if a > 0]
    
    sudoku_solution.sort()
     
    label = np.reshape(np.array(sudoku_solution*n_splits), 
                       (n_splits, len(sudoku_solution)))

    df = pd.DataFrame(label, columns = [x for x in range(len(sudoku_solution))])
    
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

def fixData(df_x, df_y): #Not finished.
    """not working"""
    
    x2y = []
    unmatched_x = []
    unmatched_y = df_y.index.tolist()
    for x_idx, x_row in enumerate(df_x):
        match = False
        for y_idx in unmatched_y:
            if ((x_row == df_y.loc[y_idx]) | (x_row == 0)).all():
                match = True
                break
        if match:
            unmatched_y.remove(y_idx)   
        else:
            x2y.append((x_idx, y_idx))
            df_y.loc[x_idx] = df_y.loc[y_idx]
            unmatched_x.append(x_idx)
    return False #df_x, df_y

def generateSynteticData(percentage = .6, path = '../data/SplitsLabel.csv', save = True):
    """"Synteticly generate incomplete sudokus out of complete sudokus"""
    
    df_syntetic = pd.read_csv(path)
    df_syntetic.reset_index(inplace = True, drop = True)
    
    nan_idx = df_syntetic[1].index[df_syntetic[1].apply(np.isnan)] 
    if len(nan_idx) > 0:
        df_syntetic.drop(labels = nan_idx, axis = 0, inplace = True)
        df_syntetic.to_csv(path)
    
    for col in range(81):
        rand_idx = np.random.randint(0, len(df_syntetic), int(len(df_syntetic)*percentage))
        rand_idx.sort()
        for r in rand_idx:
            df_syntetic.loc[r][col] = 0
            
    if save:
        name = "Syntetic"
        path = path[-10:] + name + ".csv"
        df_syntetic.to_csv(path)
    
    return df_syntetic
    
    

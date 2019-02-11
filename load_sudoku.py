# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:55:30 2019

@author: Victor Zuanazzi
"""

def load_sudoku(path, sdk_n, save = True):
    """load a txt file into a DIMAC format.
    input:
        path: path to the file with sudokus in str format
        sdk_n = number of the sudoku (= number of the line in the file) to be 
        loaded.
    return: list of list of tuples
        [[(clause (type int), negation (type boolean))]]
    """
    
    #load file
    line = str()
    with open(path, 'r') as file:
        for i in range(sdk_n+1):
            line = file.readline()
            
    print("line:" , line)
    
    sudoku = []
    str_sdk = str()
    eoc = " 0\n" #end of clause
    c = 0
    for i in range(1,10):
        for j in range(1,10):
            if line[c] != ".":
                sudoku.append(str(i)+str(j)+line[c])
                str_sdk += sudoku[-1]+eoc
            c += 1
       
    
    if save:
        name = "sudoku"+str(sdk_n)+".txt"
        save_path = ".\\data\\dimac_sudoku\\"
        with open(save_path + name, "w+") as file:
            file.write(str_sdk)

    return sudoku

if __name__ == "__main__":
    path = ".\\data\\top2365.sdk.txt"
    sdk = load_sudoku(path,
                      10)
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:46:40 2019

@author: Victor Zuanazzi
"""

def load_all_sudoku(path, name):
    """load a txt file into a DIMAC format.
    input:
        path: path to the file with sudokus in str format
        name: name of the sudoku series.
    save:
        one .txt for each line in the file. The file is in the cnf format.
    """
    
    
    line = str()
    
    with open(path, 'r') as file:
        #load file line by line
        for l, line in enumerate(file):
            
            #stores sudoku in the snf format
            str_sdk = str() 
            eoc = " 0\n" #end of clause
            
            c = 0
            for i in range(1,10):
                for j in range(1,10):
                    if line[c] != ".":
                        
                        #convertion to cnf format
                        str_sdk += str(i)+str(j)+line[c]+eoc
                    c += 1
            
            #unique name per sudoku
            s_name = name+"_"+str(l)+".txt"
            save_path = ".\\data\\dimac_sudoku\\"
            
            #save sudoku as .txt
            with open(save_path + s_name, "w+") as file:
                file.write(str_sdk)


if __name__ == "__main__":
    path = ".\\data\\damnhard.sdk.txt"
    name = "damnhard"
    sdk = load_all_sudoku(path, name)
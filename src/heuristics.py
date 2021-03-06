# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:04:17 2019

@author: Victor Zuanazzi
"""

import numpy as np
import random
from collections import Counter

def nextLiteral(cnf):
    """returns the first literal in the first clause.
    """
    return list(cnf[0].keys())[0]

def randomChoice(cnf):
    num_clauses = len(cnf)-1
    clause = np.random.randint(0, num_clauses)
    rand_lit = random.choice(list(cnf[clause].keys()))
    return rand_lit
    
def DLIS_max(cnf):
    return DLIS(cnf, take = "max")

def DLIS(cnf, take = "min"):
    """returns the literal that appears in most clauses
    """
    
    #inefficient implementation
    literal_count = {}
    for clause in cnf:
        for literal in clause:
            literal_count[literal] = literal_count.get(literal, 0) + 1
    
    if take == "max":
        #max performs terribly 
        pop_literal = max(literal_count, key=lambda key: literal_count[key])
    else:
        #min performs better
        pop_literal = min(literal_count, key=lambda key: literal_count[key])
        
    return pop_literal

def BOHM(cnf):
    """satisfy or reduce size of many preferably short clauses

    best heuristic of 1992!
    described: https://www.tcs.cs.tu-bs.de/documents/albert_schimpf_bachelors_thesis.pdf
    https://baldur.iti.kit.edu/sat/files/l05.pdf
    https://books.google.nl/books?id=5spuCQAAQBAJ&pg=PA66&lpg=PA66&dq=BOHM%E2%80%99s+Heuristic&source=bl&ots=LZW8LyS_UO&sig=ACfU3U3c80aM_2CGQgfXeD6Q3BmccS3CXg&hl=en&sa=X&ved=2ahUKEwjqqLLKlcPgAhUDfFAKHUDWCekQ6AEwBHoECAYQAQ#v=onepage&q=BOHM%E2%80%99s%20Heuristic&f=false
    """
    #store the score of each literal
    literal_score = {}
    
    #hyper-parameters suggested to be set to those values
    alpha = 1
    beta = 2
    
    #counter of literals per clause length
    len_lit_count = {}
    literals = set()

    for clause in cnf:
        l = len(clause)
        if(not l in len_lit_count):
            len_lit_count[l] = Counter()
        for literal in clause:
            len_lit_count[l][literal] += 1
            literals.add(literal)

    literal_score = []

    #Loop over all literals
    for literal in literals:
        vector = []
        
        #Loop over all length of clauses:
        for l in len_lit_count.keys():
            score = alpha*max(len_lit_count[l][literal], len_lit_count[l][-literal]) 
            score += beta*min(len_lit_count[l][literal], len_lit_count[l][-literal])
            vector.append(score)
        
        literal_score.append( (literal, vector) )

    #returns the literal that is not dominated by any other. 
    literal_score.sort(key= lambda item: item[1], reverse=True)
    bohms_favorite = literal_score[0][0]
    
    return bohms_favorite

def paretoDominant(cnf):
    """satisfy or reduce size of many preferably short clauses, based on BOHM.
   """
    #store the score of each literal
    literal_score = {}
    
    #dictionary that stores the clauses indexed in their length
    len_clauses = {}
    
    #hyperparameters suggested to be set to those values
    alpha = 1
    beta = 2
    
    #makes the dictionary that stores the clauses indexed in their length
    for clause in cnf:
        if len_clauses.get(len(clause), True):
            len_clauses[len(clause)] = []
        len_clauses[len(clause)].append(clause)                
    
    #Loop over all literals
    for clause in cnf:
        for literal in clause:
            
            #negative literals are evaluated together with positive literals
            if literal < 0: 
                continue

            vector = []
            
            #Loop over all length of clauses:
            for lc in len_clauses:
                pos_count = 0
                neg_count = 0
                
                #for every literal in the every clause, check if it is the 
                #same as the literal of interest
                for c in len_clauses[lc]:
                    for lit in c:
                        if lit == literal:
                            pos_count += 1
                        if lit == -literal:
                            neg_count += 1
                            
                vector.append(alpha*max(pos_count, neg_count) + 
                         beta*min(pos_count, neg_count))
            
            #unsure of implementation
            literal_score[literal] = np.linalg.norm(np.array(vector))
    
    #returns the literal that is not dominated by any other. 
    pareto =  max(literal_score, key=lambda key: literal_score[key])
    
    return pareto

        
    
























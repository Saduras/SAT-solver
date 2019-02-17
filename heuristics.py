# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:04:17 2019

@author: Victor Zuanazzi
"""

def popularLiteral(cnf):
    """returns the literal that appears in most clauses
    """
    
    #ineficient implementation
    literal_count = {}
    for clause in cnf:
        for literal in clause:
            literal_count[literal] = literal_count.get(literal, 0) + 1
            
    return max(literal_count, key=lambda key: literal_count[key])


def clause_killer(cnf):
    """returns the literal that solves one clause (and possibly the assigment)
    """
    pass
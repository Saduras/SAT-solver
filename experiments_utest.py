# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:51:29 2019

@author: Victor Zuanazzi
"""

import unittest
from collections import defaultdict
from experiments import removeTautology, removeUnitClause, assign, split, DP
from heuristics import DLIS, BOHM, nextLiteral, randomChoice

class DPTests(unittest.TestCase):
    def test_removeTautology_empty(self):
        cnf = []
        self.assertEqual(removeTautology(cnf), cnf)

    def test_removeTautology_nochange(self):
        cnf = [{-123:True, 113:True},{321,True}]
        self.assertEqual(removeTautology(cnf), cnf)

    def test_removeTautology_inconsitent(self):
        cnf = [{-123:True, 113:True},{321,True}, {}]
        self.assertEqual(removeTautology(cnf), cnf)

    def test_removeTautology_simple(self):
        cnf = [{-123:True, 123:True}]
        self.assertEqual(removeTautology(cnf),[])

    def test_removeTautology_with_noise(self):
        cnf = [{-123:True, -312:True, 423:True, 123:True}]
        self.assertEqual(removeTautology(cnf),[])

    def test_removeUnitClause_empty(self):
        cnf = []
        assignment = []
        stats = {"unit_clause_calls": 0, "unit_clause_time": 0}

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                                  assignment,
                                                                  stats)
        self.assertEqual(new_cnf, cnf)
        self.assertEqual(new_assignment, assignment)
        self.assertEqual(change, False)

    def test_removeUnitClause_noUnit(self):
        cnf = [{-123:True, 321:True}]
        assignment = []
        stats = {"unit_clause_calls": 0, "unit_clause_time": 0}

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                                  assignment,
                                                                  stats)
        self.assertEqual(new_cnf, cnf)
        self.assertEqual(new_assignment, assignment)
        self.assertEqual(change, False)

    def test_removeUnitClause_onlyNegUnit(self):
        cnf = [{-123:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                                  assignment,
                                                                  stats)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_removeUnitClause_onlyPosUnit(self):
        cnf = [{123:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                           assignment,
                                                           stats)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [123])
        self.assertEqual(change, True)

    def test_removeUnitClause_oneUnitAndNoise(self):
        cnf = [{-123:True}, {423:True, -352:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                           assignment,
                                                           stats)
        self.assertEqual(new_cnf, [{423:True, -352:True}])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_removeUnitClause_manyUnit(self):
        cnf = [{-123:True}, {234:True}, {423:True, -352:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                                  assignment,
                                                                  stats)
        self.assertEqual(new_cnf, [{423:True, -352:True}])
        self.assertEqual(new_assignment, [-123, 234])
        self.assertEqual(change, True)
    
    def test_removeUnitClause_applyAssignment(self):
        cnf = [{-123:True}, {123:True, -352:True, 423:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                                  assignment,
                                                                  stats)
        self.assertEqual(new_cnf, [{-352:True, 423:True}])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_removeUnitClause_copyClauses(self):
        cnf = [{-123:True}, {-123:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }

        new_cnf, new_assignment, change, stats = removeUnitClause(cnf, 
                                                                  assignment,
                                                                  stats)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_assign_oneOccurencePos(self):
        cnf = [{123:True}]
        assignment = []
        stats = {"assign_calls": 0, "assign_time": 0}

        new_cnf, new_assignment, stats = assign(123, 
                                               True, 
                                               cnf, 
                                               assignment, 
                                               stats)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [123])

    def test_assign_oneOccurenceNeg(self):
        cnf = [{123:True}]
        assignment = []
        stats = {"assign_calls": 0, "assign_time": 0}

        new_cnf, new_assignment, stats = assign(123, 
                                                False,
                                                cnf, 
                                                assignment,
                                                stats)
        self.assertEqual(new_cnf, [{}])
        self.assertEqual(new_assignment, [-123])

    def test_assign_mutliOccurenceSameSign(self):
        cnf = [{123:True},{123:True},{-234:True}]
        assignment = []
        stats = {"assign_calls": 0, "assign_time": 0}

        new_cnf, new_assignment, stats = assign(123, 
                                                True, 
                                                cnf, 
                                                assignment,
                                                stats)
        
        self.assertEqual(new_cnf, [ {-234:True} ])
        self.assertEqual(new_assignment, [123])
    
    def test_assign_mutliOccurenceMixedSign(self):
        cnf = [{123:True, -123:True}, {-123:True}, {-234:True}]
        assignment = []
        stats = {"assign_calls": 0, "assign_time": 0}

        new_cnf, new_assignment, stats = assign(123, 
                                                True, 
                                                cnf, 
                                                assignment,
                                                stats)
        self.assertEqual(new_cnf, [ {}, {-234:True} ])
        self.assertEqual(new_assignment, [123])

    def test_assign_removeLiterals(self):
        cnf = [{123:True, -123:True}, {-123:True, -234:True}]
        assignment = []
        stats = {"assign_calls": 0, "assign_time": 0}

        new_cnf, _, _ = assign(123, True, cnf, assignment, stats)
        self.assertEqual(cnf, [{123:True, -123:True}, {-123:True, -234:True}])
        self.assertEqual(new_cnf, [ {-234:True} ])

    def test_split_oneClausePos(self):
        cnf = [{123: True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }
        heuristic = "next"

        new_cnf, new_assignment, stats = split(True, 
                                               cnf, 
                                               assignment,
                                               heuristic,
                                               stats)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [123])

    def test_split_oneClauseNeg(self):
        cnf = [{123:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }
        heuristic = "next"

        new_cnf, new_assignment, stats = split(False, 
                                               cnf, 
                                               assignment,
                                               heuristic,
                                               stats)
        self.assertEqual(new_cnf, [{}])
        self.assertEqual(new_assignment, [-123])

    def test_split_multiClause(self):
        cnf = [{123:True}, {234:True}]
        assignment = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }
        heuristic = "next"

        new_cnf, new_assignment, stats = split(True, 
                                               cnf, 
                                               assignment,
                                               heuristic,
                                               stats)
        
        self.assertEqual(new_cnf, [{234:True}])
        self.assertEqual(new_assignment, [123])

    def test_split_empty(self):
        cnf = []
        assignment = []
        stats = {"split_calls": 0, "split_time": 0}
        heuristic = "next"

        with self.assertRaises(Exception) as context:
            split(True, cnf, assignment, heuristic, stats)

        self.assertTrue('Invalid CNF' in str(context.exception))

    def test_DP_empty(self):
        cnf = []
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }
        heuristic = "next"

        assignment, stats = DP(cnf, heuristic, stats)
        self.assertEqual(assignment, [])

    def test_DP_tinyCase(self):
        cnf = [{123:True}]
        stats = {
            "DP_calls": 0,
            "split_calls": 0,
            "split_time": 0,
            "assign_calls": 0,
            "assign_time": 0,
            "unit_clause_calls": 0,
            "unit_clause_time": 0
            }
        heuristic = "next"

        assignment, stats = DP(cnf, heuristic, stats)
        self.assertEqual(assignment, [123])
        
    def test_DLIS_trivial(self):
        cnf = [{-123:True}]
        
        literal = DLIS(cnf)
        self.assertEqual(literal, (-123, True))
    
    def test_DLIS_tie(self):
        cnf = [{-123:True, 234:True}]
        
        literal = DLIS(cnf)
        self.assertEqual(literal, (-123, True))

    def test_DLIS_twoClause(self):
        cnf = [{-123:True, -312:True, 423:True},
               {-123:True, 423:True}]
        
        literal = DLIS(cnf)
        self.assertEqual(literal, (-312, True))
    
    def test_BOHM_trivial(self):
        cnf = [{-123:True}]
        
        literal = BOHM(cnf)
        self.assertEqual(literal, (-123, True))

    def test_BOHM_multiClauseTie(self):
        cnf = [{-123:True, 234:True}, {123:True, -234:True, 456:True}]
        
        literal = BOHM(cnf)
        self.assertNotEqual(literal, (456, True))

    def test_BOHM_shortClausePrio(self):
        cnf = [{-123:True, 234:True, 345:True}, {345:True}, {234:True, 567:True}]
        
        literal = BOHM(cnf)
        self.assertEqual(literal, (345, True))
        
    def test_nextLiteral(self):
        cnf = [{-123:True, -312:True, 423:True, 123:True},
               {-123:True}]
        
        literal = nextLiteral(cnf)
        self.assertEqual(literal, (-123, True))
        
    

if __name__ == '__main__':
    unittest.main()
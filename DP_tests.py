import unittest
from collections import defaultdict
from DP import removeTautology, removeUnitClause, assign, split, DP
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

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, cnf)
        self.assertEqual(new_assignment, assignment)
        self.assertEqual(change, False)

    def test_removeUnitClause_noUnit(self):
        cnf = [{-123:True, 321:True}]
        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, cnf)
        self.assertEqual(new_assignment, assignment)
        self.assertEqual(change, False)

    def test_removeUnitClause_onlyNegUnit(self):
        cnf = [{-123:True}]
        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_removeUnitClause_onlyPosUnit(self):
        cnf = [{123:True}]
        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [123])
        self.assertEqual(change, True)

    def test_removeUnitClause_oneUnitAndNoise(self):
        cnf = [{-123:True}, {423:True, -352:True}]
        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, [{423:True, -352:True}])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_removeUnitClause_manyUnit(self):
        cnf = [{-123:True}, {234:True}, {423:True, -352:True}]
        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, [{423:True, -352:True}])
        self.assertEqual(new_assignment, [-123, 234])
        self.assertEqual(change, True)
    
    def test_removeUnitClause_applyAssignment(self):
        cnf = [{-123:True}, {123:True, -352:True, 423:True}]
        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, [{-352:True, 423:True}])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_removeUnitClause_copyClauses(self):
        cnf = [{-123:True}, {-123:True}]

        assignment = []

        new_cnf, new_assignment, change = removeUnitClause(cnf, assignment)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [-123])
        self.assertEqual(change, True)

    def test_assign_oneOccurencePos(self):
        cnf = [{123:True}]
        assignment = []

        new_cnf, new_assignment = assign(123, True, cnf, assignment)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [123])

    def test_assign_oneOccurenceNeg(self):
        cnf = [{123:True}]
        assignment = []

        new_cnf, new_assignment = assign(123, False, cnf, assignment)
        self.assertEqual(new_cnf, [{}])
        self.assertEqual(new_assignment, [-123])

    def test_assign_mutliOccurenceSameSign(self):
        cnf = [{123:True},{123:True},{-234:True}]
        assignment = []

        new_cnf, new_assignment = assign(123, True, cnf, assignment)
        self.assertEqual(new_cnf, [ {-234:True} ])
        self.assertEqual(new_assignment, [123])
    
    def test_assign_mutliOccurenceMixedSign(self):
        cnf = [{123:True, -123:True}, {-123:True}, {-234:True}]
        assignment = []

        new_cnf, new_assignment = assign(123, True, cnf, assignment)
        self.assertEqual(new_cnf, [ {}, {-234:True} ])
        self.assertEqual(new_assignment, [123])

    def test_assign_removeLiterals(self):
        cnf = [{123:True, -123:True}, {-123:True, -234:True}]
        assignment = []

        new_cnf, _ = assign(123, True, cnf, assignment)
        self.assertEqual(cnf, [{123:True, -123:True}, {-123:True, -234:True}])
        self.assertEqual(new_cnf, [ {-234:True} ])

    def test_split_oneClausePos(self):
        cnf = [{123: True}]
        assignment = []

        new_cnf, new_assignment = split(True, cnf, assignment)
        self.assertEqual(new_cnf, [])
        self.assertEqual(new_assignment, [123])

    def test_split_oneClauseNeg(self):
        cnf = [{123:True}]
        assignment = []

        new_cnf, new_assignment = split(False, cnf, assignment)
        self.assertEqual(new_cnf, [{}])
        self.assertEqual(new_assignment, [-123])

    def test_split_multiClause(self):
        cnf = [{123:True}, {234:True}]
        assignment = []

        new_cnf, new_assignment = split(True, cnf, assignment)
        self.assertEqual(new_cnf, [{234:True}])
        self.assertEqual(new_assignment, [123])

    def test_split_empty(self):
        cnf = []
        assignment = []

        with self.assertRaises(Exception) as context:
            split(True, cnf, assignment)

        self.assertTrue('Invalid CNF' in str(context.exception))

    def test_DP_empty(self):
        cnf = []

        assignment = DP(cnf)
        self.assertEqual(assignment, [])

    def test_DP_tinyCase(self):
        cnf = [{123:True}]

        assignment = DP(cnf)
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
        cnf = [{-123:True, -312:True, 423:True, 123:True},
               {-123:True}]
        
        literal = DLIS(cnf)
        self.assertEqual(literal, (-123, True))
        
    def test_nextLiteral(self):
        cnf = [{-123:True, -312:True, 423:True, 123:True},
               {-123:True}]
        
        literal = nextLiteral(cnf)
        self.assertEqual(literal, (-123, True))
        
    

if __name__ == '__main__':
    unittest.main()
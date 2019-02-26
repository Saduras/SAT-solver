import unittest
from heuristics import DLIS, BOHM, nextLiteral, randomChoice

class DPTests(unittest.TestCase):
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
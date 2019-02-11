import unittest
from DP import removeTautology

class DPTests(unittest.TestCase):
    def test_removeTautology_empty(self):
        cnf = []
        self.assertEqual(removeTautology(cnf), cnf)

    def test_removeTautology_nochange(self):
        cnf = [[(123, True), (113, False)],[(321, False)]]
        self.assertEqual(removeTautology(cnf), cnf)

    def test_removeTautology_inconsitent(self):
        cnf = [[(123, True), (113, False)],[(321, False)], []]
        self.assertEqual(removeTautology(cnf), cnf)

    def test_removeTautology_simple(self):
        cnf = [[(123, True), (123, False)]]
        self.assertEqual(removeTautology(cnf),[])

    def test_removeTautology_with_noise(self):
        cnf = [[(123, True), (312, True), (423, False), (123, False)]]
        self.assertEqual(removeTautology(cnf),[])

if __name__ == '__main__':
    unittest.main()
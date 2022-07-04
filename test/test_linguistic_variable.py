import unittest
from fuzzy_inference.fuzzy_set import FuzzySet
from fuzzy_inference.linguistic_variable import LinguisticVariable


class LinguisticVariableTestCase(unittest.TestCase):

    def test_linvariable_crisp_measurement(self):
        quantifier = LinguisticVariable('Quantifier', terms={
            'Few': FuzzySet.l_ramp(0.2, 0.4),
            'About Half': FuzzySet.trapezoidal(0.2, 0.4, 0.6, 0.8),
            'Most': FuzzySet.r_ramp(0.6, 0.8)
        })
        self.assertEqual(quantifier.fuzzify_crisp(0.23)['Few'], 0.85)

    def test_linvariable_fuzzy_measurement(self):
        quantifier = LinguisticVariable('Quantifier', terms={
            'Few': FuzzySet.l_ramp(0.2, 0.4),
            'About Half': FuzzySet.trapezoidal(0.2, 0.4, 0.6, 0.8),
            'Most': FuzzySet.r_ramp(0.6, 0.8)
        })
        self.assertAlmostEqual(quantifier.fuzzify_fuzzy(
            FuzzySet.triangular(0.3, 0.4, 0.5))['Few'], 0.333, 3)

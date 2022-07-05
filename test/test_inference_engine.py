import unittest
from fuzzy_inference.fuzzy_set import FuzzySet

from fuzzy_inference.inference_engine import InferenceEngine
from fuzzy_inference.linguistic_variable import LinguisticVariable
from fuzzy_inference.rule import Rule



class InferenceEngineTestCase(unittest.TestCase):

    def test_basic(self):
        engine = InferenceEngine()
        engine.inputvars = [
            LinguisticVariable('wealth', {
                'poor': FuzzySet.l_ramp(10000, 25000),
                'middle': FuzzySet.trapezoidal(10000, 25000, 50000, 65000),
                'rich': FuzzySet.r_ramp(50000, 65000)
            }, [0, 100_000]),
            LinguisticVariable('age', {
                'young': FuzzySet.l_ramp(35, 50),
                'middle': FuzzySet.triangular(35, 50, 60),
                'old': FuzzySet.r_ramp(50, 60)
            }, [0, 150]),

        ]
        engine.outputvars = [
            LinguisticVariable('risk', {
                'low': FuzzySet.l_ramp(0.2, 0.4),
                'medium': FuzzySet.trapezoidal(0.2, 0.4, 0.6, 0.8),
                'high': FuzzySet.r_ramp(0.6, 0.8)
            }, [0, 1])
        ]

        engine._rulebase = [
            Rule().IF(
                (
                    ('age', 'young'),
                    ('wealth', 'rich')
                )
            ).THEN(
                ('risk', 'low')
            )
        ]

        engine.infer({
            'wealth': FuzzySet.triangular(50000, 60000, 70000),
            'age': 25
        })


        results = engine.defuzzify()
        self.assertTrue(0.0 < results['risk'] < 0.4)

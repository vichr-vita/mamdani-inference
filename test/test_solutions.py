import unittest

from fuzzy_inference.fuzzy_set import FuzzySet
from fuzzy_inference.inference_engine import InferenceEngine
from fuzzy_inference.linguistic_variable import LinguisticVariable
from fuzzy_inference.rule import Rule


class SolutionsTestCase(unittest.TestCase):

    def test_summary_inference(self):
        obligation_adherence = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

        at_most_few = FuzzySet.l_ramp(3, 8)
        number_of_late_payments = len(
            [x for x in obligation_adherence if not x])
        validity = at_most_few.mu(number_of_late_payments)

        # take into account questionnaire and hard data
        late_payments = LinguisticVariable('no. late payments', {
            'at most few': FuzzySet.l_ramp(3, 8),
            'more than few': FuzzySet.r_ramp(3, 8)
        })
        conscientiousness = LinguisticVariable('conscientiousness', {
            'low': FuzzySet.l_ramp(0.2, 0.3),
            'moderate': FuzzySet.trapezoidal(0.2, 0.4, 0.6, 0.8),
            'high': FuzzySet.r_ramp(0.6, 0.8)
        }, (0, 1), 'percentile')

        engine = InferenceEngine()
        engine.inputvars = [late_payments, conscientiousness]
        engine.outputvars = [conscientiousness]
        engine.rulebase = [
            Rule().IF(
                (
                    ('conscientiousness', 'high'),
                    ('no. late payments', 'at most few')
                )
            ).THEN(
                ('conscientiousness', 'high')
            ),
            Rule().IF(
                (
                    ('conscientiousness', 'high'),
                    ('no. late payments', 'more than few')
                )
            ).THEN(
                ('conscientiousness', 'low')
            ),
            Rule().IF(
                (
                    ('conscientiousness', 'moderate'),
                    ('no. late payments', 'more than few')
                )
            ).THEN(
                ('conscientiousness', 'low')
            ),
            Rule().IF(
                (
                    ('conscientiousness', 'low'),
                    ('no. late payments', 'more than few')
                )
            ).THEN(
                ('conscientiousness', 'low')
            )
        ]

        engine.infer({
            'no. late payments': FuzzySet.triangular(7, 7, 8),
            'conscientiousness': FuzzySet.triangular(0.8, 0.85, 0.9)
        })

        results_2 = engine.defuzzify()

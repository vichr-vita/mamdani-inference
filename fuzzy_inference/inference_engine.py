from fuzzy_inference.fuzzy_set import DEFAULT_RESOLUTION, FuzzySet
from fuzzy_inference.linguistic_variable import LinguisticVariable
from fuzzy_inference.rule import Rule
import numpy as np
import math
import matplotlib.pyplot as plt


class InferenceEngine:

    def __init__(self) -> None:
        self._inputvars: dict[str, LinguisticVariable] = {}
        self._outputvars: dict[str, LinguisticVariable] = {}
        self._rulebase: list[Rule] = []

    @property
    def rulebase(self):
        return self._rulebase

    @rulebase.setter
    def rulebase(self, rulebase: list[Rule]) -> None:
        self._rulebase = rulebase

    @property
    def inputvars(self) -> dict[str, LinguisticVariable]:
        return self._inputvars

    @inputvars.setter
    def inputvars(self, inputvars: list[LinguisticVariable]) -> None:
        self._inputvars = {l.name: l for l in inputvars}

    @property
    def outputvars(self) -> dict[str, LinguisticVariable]:
        return self._outputvars

    @outputvars.setter
    def outputvars(self, outputvars: list[LinguisticVariable]) -> None:
        self._outputvars = {l.name: l for l in outputvars}

    def infer(self, measurements: dict):
        for k, v in measurements.items():
            self.inputvars[k].value = v

        for rule in self._rulebase:
            if rule.consequent is None:
                raise ValueError('consequent cannot be none at this stage.')
            ceil: float = min([self.inputvars[ant[0]].fuzzified[ant[1]]
                               for ant in rule.antecedents])
            outvar: LinguisticVariable = self.outputvars[rule.consequent[0]]
            # you do not actually need fuzzified, you need the initial fuzzy set

            B = outvar.terms[rule.consequent[1]]
            B_prime = FuzzySet.intersection(
                B, FuzzySet.uniform(ceil), (outvar.min, outvar.max))
            outvar.b_prime[rule.consequent[1]] = B_prime

    def output_fuzzy(self, resolution: int = DEFAULT_RESOLUTION) -> dict[str, FuzzySet]:
        results = {}
        for var_name, var in self.outputvars.items():
            fuzz = FuzzySet.uniform(0)
            for term_name, term in var.b_prime.items():
                fuzz = FuzzySet.union(
                    fuzz, term, (var.min, var.max), resolution=DEFAULT_RESOLUTION)
                # I need to perform the intesection operation on ALL the terms, not just one
                results[var_name] = fuzz
        return results


    def defuzzify(self):
        """uses center of maxima function to defuzzify output measures"""
        results = {}
        for var_name, var in self.outputvars.items():
            fuzz = FuzzySet.uniform(0)
            for term_name, term in var.b_prime.items():
                fuzz = FuzzySet.union(fuzz, term, (var.min, var.max), resolution=DEFAULT_RESOLUTION)
                # I need to perform the intesection operation on ALL the terms, not just one
            maximum = np.max(fuzz.vertices[:,1])
            maxima = np.where(fuzz.vertices[:,1] == maximum)
            max_left, max_right = maxima[0][0], maxima[0][-1]
            x_max_index = (max_left + max_right)/2
            lower_i = fuzz.vertices[math.floor(x_max_index)]
            upper_i = fuzz.vertices[math.ceil(x_max_index)]
            defuzz_x = (upper_i[0] + lower_i[0])/2
            results[var_name] = defuzz_x
        return results
        
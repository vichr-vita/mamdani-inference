from __future__ import annotations
from typing import Collection

from fuzzy_inference.fuzzy_set import FuzzySet

import matplotlib.pyplot as plt


class LinguisticVariable:
    def __init__(self, name, terms: dict[str, FuzzySet], range: Collection = (0, 1), domain: str = 'x') -> None:
        self.name = name
        self.terms: dict[str, FuzzySet] = terms
        self._a_prime: float | FuzzySet = 0
        self.min, self.max = range
        self.fuzzified: dict[str, float] = {}  # for inputs
        self.b_prime: dict[str, FuzzySet] = {k: FuzzySet.uniform(
            0) for k, _ in self.terms.items()}  # for fuzzy outputs
        self.domain = domain

    @property
    def a_prime(self):
        return self._a_prime

    @a_prime.setter
    def value(self, value: int | float | FuzzySet) -> None:
        self._a_prime = value
        if type(value) == float or type(value) == int:
            self.fuzzified = self.fuzzify_crisp(self._a_prime)  # type: ignore
        elif type(value) == FuzzySet:
            self.fuzzified = self.fuzzify_fuzzy(self._a_prime)  # type: ignore
        else:
            raise ValueError(str(type(value)) + ' is not supported.')

    def fuzzify_crisp(self, x: float) -> dict[str, float]:
        fuz = {}
        for k, v in self.terms.items():
            fuz[k] = v.mu(x)
        return fuz

    def fuzzify_fuzzy(self, x: FuzzySet) -> dict[str, float]:
        fuz = {}
        for k, v in self.terms.items():
            fuz[k] = FuzzySet.intersection(
                v, x, (self.min, self.max)).height
        return fuz
